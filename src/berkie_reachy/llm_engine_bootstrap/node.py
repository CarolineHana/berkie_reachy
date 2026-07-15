"""Detect Node.js/Yarn and manage the llm_engine Node service as a subprocess.

Node.js, Yarn, and MongoDB are system-level runtimes that cannot be silently
installed on someone's machine without their consent - unlike ``chromadb``
(a same-venv pip install), these get detect-and-guide handling only.
"""

from __future__ import annotations

import re
import shutil
import signal
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from berkie_reachy.llm_engine_bootstrap import state

logger = logging.getLogger(__name__)

PROCESS_NAME = "llm_engine"
SUPPORTED_NODE_MAJORS = (20, 22)
PINNED_YARN_VERSION = "1.22.22"

LLM_ENGINE_URL = f"http://127.0.0.1:{state.LLM_ENGINE_PORT}"


@dataclass
class DetectionResult:
    """Result of checking whether Node/Yarn are usable on this host."""

    node_found: bool
    node_version: Optional[str]
    node_supported: bool
    yarn_found: bool
    instructions: Optional[str] = None


def detect_node_and_yarn() -> DetectionResult:
    """Check for a usable Node.js + Yarn on this host, without installing anything."""
    node_path = shutil.which("node")
    if node_path is None:
        return DetectionResult(
            node_found=False,
            node_version=None,
            node_supported=False,
            yarn_found=False,
            instructions=(
                "Node.js was not found. Install Node.js 20 or 22 from https://nodejs.org "
                "(or via your system package manager), then relaunch the app."
            ),
        )

    try:
        raw = subprocess.run([node_path, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        raw = ""
    match = re.match(r"v(\d+)\.", raw)
    major = int(match.group(1)) if match else None
    supported = major in SUPPORTED_NODE_MAJORS

    yarn_found = shutil.which("yarn") is not None or _corepack_available()

    instructions = None
    if not supported:
        instructions = (
            f"Node.js {raw or '(unknown version)'} was found, but llm_engine needs Node 20 or 22. "
            "Install a supported version from https://nodejs.org, then relaunch the app."
        )
    elif not yarn_found:
        instructions = (
            "Yarn was not found and Corepack is unavailable. Install Yarn "
            f"(`npm install -g yarn` or enable Corepack: `corepack enable`), then relaunch the app."
        )

    return DetectionResult(
        node_found=True,
        node_version=raw,
        node_supported=supported,
        yarn_found=yarn_found,
        instructions=instructions,
    )


def _corepack_available() -> bool:
    return shutil.which("corepack") is not None


def ensure_yarn_ready() -> str:
    """Return a working ``yarn`` command, activating the pinned version via Corepack if needed."""
    yarn_path = shutil.which("yarn")
    if yarn_path:
        return yarn_path

    if _corepack_available():
        subprocess.run(["corepack", "enable"], check=False, capture_output=True)
        subprocess.run(
            ["corepack", "prepare", f"yarn@{PINNED_YARN_VERSION}", "--activate"],
            check=True,
            capture_output=True,
            text=True,
        )
        yarn_path = shutil.which("yarn")
        if yarn_path:
            return yarn_path

    raise RuntimeError("Yarn is not available and could not be activated via Corepack.")


def ensure_dependencies_installed(src_dir: Path, yarn_cmd: str) -> None:
    """Run `yarn install` if node_modules is missing (first run on this host)."""
    node_modules = src_dir / "node_modules"
    if node_modules.exists():
        logger.debug("node_modules already present, skipping yarn install")
        return
    logger.info("Installing llm_engine dependencies (this can take a few minutes on first run)...")
    subprocess.run([yarn_cmd, "install", "--frozen-lockfile"], cwd=str(src_dir), check=True)


def ensure_built(src_dir: Path, yarn_cmd: str) -> None:
    """Run `yarn build` (tsc) if the compiled entrypoint is missing."""
    entrypoint = src_dir / "dist" / "src" / "index.js"
    if entrypoint.exists():
        logger.debug("llm_engine already built, skipping yarn build")
        return
    logger.info("Building llm_engine...")
    subprocess.run([yarn_cmd, "build"], cwd=str(src_dir), check=True)


def is_llm_engine_healthy() -> bool:
    """Check llm_engine's /v1/health endpoint."""
    try:
        import httpx

        resp = httpx.get(f"{LLM_ENGINE_URL}/v1/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_llm_engine_running(src_dir: Path, env: dict[str, str], *, timeout: float = 60.0) -> Optional[int]:
    """Start the llm_engine Node service if not already healthy. Returns the started pid, or None if reused."""
    if is_llm_engine_healthy():
        logger.info("Reusing already-running llm_engine on port %s", state.LLM_ENGINE_PORT)
        return None

    entrypoint = src_dir / "dist" / "src" / "index.js"
    if not entrypoint.exists():
        raise FileNotFoundError(f"llm_engine is not built yet: {entrypoint} missing")

    state.ensure_dirs()
    logfile = state.LOGS_DIR / "llm_engine.log"
    with open(logfile, "wb") as log_f:
        proc = subprocess.Popen(
            ["node", "dist/src/index.js"],
            cwd=str(src_dir),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_llm_engine_healthy():
            state.write_pidfile(PROCESS_NAME, proc.pid)
            logger.info("Started llm_engine (pid %s) on port %s", proc.pid, state.LLM_ENGINE_PORT)
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"llm_engine exited early (code {proc.returncode}); see {logfile}")
        time.sleep(0.5)

    proc.terminate()
    raise TimeoutError(f"llm_engine did not become healthy within {timeout}s; see {logfile}")


def stop_llm_engine_if_started_by_us() -> None:
    """Stop the llm_engine process this run started, if any."""
    pid = state.read_pidfile(PROCESS_NAME)
    if pid is None:
        return
    logger.info("Stopping llm_engine (pid %s)", pid)
    state.stop_pid(pid, term_signal=signal.SIGINT)
    state.clear_pidfile(PROCESS_NAME)
