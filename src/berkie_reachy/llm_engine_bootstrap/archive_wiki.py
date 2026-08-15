"""Clone/build/run the BKC archive-wiki API locally, so Berky's eventHistorian agent can search it.

Mirrors repo.py/node.py's pattern for llm_engine itself: the wiki content and
its small HTTP API aren't vendored into berkie_reachy - they're cloned fresh
from the public berkmancenter/bkc-archive-wiki repo onto the host machine,
built, and run as a local subprocess. Best-effort by design: if this fails to
clone/build/start, Berky's core voice loop (web_search, event_history) still
works fine without it - eventHistorian's bkcArchiveWikiTools just stays empty.
"""

from __future__ import annotations
import os
import time
import logging
import subprocess
from typing import Optional
from pathlib import Path

from berkie_reachy.llm_engine_bootstrap import state


logger = logging.getLogger(__name__)

PROCESS_NAME = "archive_wiki"
ARCHIVE_WIKI_REPO_URL = "https://github.com/berkmancenter/bkc-archive-wiki.git"
ARCHIVE_WIKI_PINNED_REF = "main"
ARCHIVE_WIKI_URL = f"http://127.0.0.1:{state.ARCHIVE_WIKI_PORT}"


class ArchiveWikiError(RuntimeError):
    """Raised when cloning/building/running the archive-wiki API fails."""


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 300) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise ArchiveWikiError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def ensure_archive_wiki_source(dest_dir: Path | None = None) -> Path:
    """Clone or update bkc-archive-wiki into ``dest_dir``. Idempotent; safe on every launch."""
    dest_dir = dest_dir or state.ARCHIVE_WIKI_SRC_DIR
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if (dest_dir / ".git").exists():
        logger.info("Updating existing bkc-archive-wiki checkout at %s", dest_dir)
        _run(["git", "fetch", "--depth", "1", "origin", ARCHIVE_WIKI_PINNED_REF], cwd=dest_dir)
        _run(["git", "reset", "--hard", f"origin/{ARCHIVE_WIKI_PINNED_REF}"], cwd=dest_dir)
    else:
        logger.info("Cloning bkc-archive-wiki into %s", dest_dir)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ARCHIVE_WIKI_PINNED_REF,
                ARCHIVE_WIKI_REPO_URL,
                str(dest_dir),
            ]
        )
    return dest_dir


def _api_dir(src_dir: Path) -> Path:
    return src_dir / "api"


def ensure_dependencies_installed(src_dir: Path) -> None:
    """Run `npm ci` in api/ if node_modules is missing or stale relative to the checked-out commit."""
    api_dir = _api_dir(src_dir)
    node_modules = api_dir / "node_modules"
    marker = node_modules / ".berky-install-commit"
    if node_modules.exists() and not state.build_is_stale(src_dir, marker):
        logger.debug("archive-wiki api node_modules already present and current, skipping npm ci")
        return
    logger.info("Installing archive-wiki API dependencies...")
    npm_path = state.find_executable("npm")
    if npm_path is None:
        raise ArchiveWikiError("npm not found (should ship with Node.js)")
    subprocess.run([npm_path, "ci"], cwd=str(api_dir), check=True, timeout=300)
    state.write_build_marker(src_dir, marker)


def ensure_built(src_dir: Path) -> None:
    """Run `npm run build` (tsc) in api/ if dist is missing or stale relative to source."""
    api_dir = _api_dir(src_dir)
    entrypoint = api_dir / "dist" / "index.js"
    marker = api_dir / "dist" / ".berky-build-commit"
    if entrypoint.exists() and not state.build_is_stale(src_dir, marker):
        logger.debug("archive-wiki api already built and current, skipping npm run build")
        return
    logger.info("Building archive-wiki API...")
    npm_path = state.find_executable("npm")
    if npm_path is None:
        raise ArchiveWikiError("npm not found (should ship with Node.js)")
    subprocess.run([npm_path, "run", "build"], cwd=str(api_dir), check=True, timeout=120)
    state.write_build_marker(src_dir, marker)


def is_archive_wiki_healthy() -> bool:
    """Check the archive-wiki API's /v1/health endpoint."""
    try:
        import httpx

        resp = httpx.get(f"{ARCHIVE_WIKI_URL}/v1/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_archive_wiki_running(src_dir: Path, *, timeout: float = 60.0) -> Optional[int]:
    """Start the archive-wiki API (production mode) if not already healthy.

    Returns the pid this call started, or None if reused.
    """
    if is_archive_wiki_healthy():
        logger.info("Reusing already-running archive-wiki API on port %s", state.ARCHIVE_WIKI_PORT)
        return None

    api_dir = _api_dir(src_dir)
    entrypoint = api_dir / "dist" / "index.js"
    if not entrypoint.exists():
        raise FileNotFoundError(f"archive-wiki API is not built yet: {entrypoint} missing")

    node_path = state.find_executable("node")
    if node_path is None:
        raise FileNotFoundError("node executable not found (was available at detection time, now missing?)")

    state.ensure_dirs()
    logfile = state.LOGS_DIR / "archive_wiki.log"
    full_env = dict(os.environ)
    full_env["PORT"] = str(state.ARCHIVE_WIKI_PORT)

    with open(logfile, "wb") as log_f:
        proc = subprocess.Popen(
            [node_path, "dist/index.js"],
            cwd=str(api_dir),
            env=full_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_archive_wiki_healthy():
            state.write_pidfile(PROCESS_NAME, proc.pid)
            logger.info("Started archive-wiki API (pid %s) on port %s", proc.pid, state.ARCHIVE_WIKI_PORT)
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"archive-wiki API exited early (code {proc.returncode}); see {logfile}")
        time.sleep(0.3)

    proc.terminate()
    raise TimeoutError(f"archive-wiki API did not become healthy within {timeout}s; see {logfile}")


def stop_archive_wiki_if_started_by_us() -> None:
    """Stop the archive-wiki API process this run started, if any."""
    pid = state.read_pidfile(PROCESS_NAME)
    if pid is None:
        return
    logger.info("Stopping archive-wiki API (pid %s)", pid)
    state.stop_pid(pid)
    state.clear_pidfile(PROCESS_NAME)


def ensure_archive_wiki_stack(*, timeout: float = 60.0) -> Optional[str]:
    """Clone/build/run the archive-wiki API end to end. Returns its URL on success, None on any failure.

    Deliberately swallows all errors (logged, not raised) - this is a
    best-effort enhancement, not a hard dependency of Berky's voice loop.
    """
    try:
        src_dir = ensure_archive_wiki_source()
        ensure_dependencies_installed(src_dir)
        ensure_built(src_dir)
        ensure_archive_wiki_running(src_dir, timeout=timeout)
        return ARCHIVE_WIKI_URL
    except Exception:
        logger.warning(
            "Could not start the local BKC archive-wiki API; eventHistorian's archive "
            "search tools will just stay unavailable. Continuing without it.",
            exc_info=True,
        )
        return None
