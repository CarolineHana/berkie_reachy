"""Detect/install/start/stop a local ChromaDB server for llm_engine's RAG features.

ChromaDB is installed into its own dedicated virtual environment
(``~/.berkie_reachy/llm_backend/chroma-venv``), NOT into berkie_reachy's own
Python environment. Installing it into the shared environment was tried
first and rejected: it pulled in a newer ``huggingface_hub``, breaking
``reachy_mini``'s pinned ``huggingface_hub==1.3.0`` (confirmed empirically -
that install left ``reachy_mini`` importable but at real risk of subtle
breakage in code paths like ``snapshot_download``, which the app-installer
itself relies on). Since Chroma runs as a separate subprocess anyway (we
never `import chromadb` from berkie_reachy's own process), there's no reason
for it to share an interpreter/environment with reachy_mini at all.
"""

from __future__ import annotations

import sys
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from berkie_reachy.llm_engine_bootstrap import state

logger = logging.getLogger(__name__)

PROCESS_NAME = "chroma"
CHROMA_DB_URL = f"http://127.0.0.1:{state.CHROMA_PORT}"

CHROMA_VENV_DIR = state.APP_DATA_ROOT / "chroma-venv"


def _venv_python() -> Path:
    return CHROMA_VENV_DIR / "bin" / "python3"


def _venv_chroma() -> Path:
    return CHROMA_VENV_DIR / "bin" / "chroma"


def is_chromadb_installed() -> bool:
    """Return True if chromadb is installed in its dedicated venv."""
    if not _venv_chroma().exists():
        return False
    result = subprocess.run(
        [str(_venv_python()), "-c", "import chromadb"],
        capture_output=True,
    )
    return result.returncode == 0


def install_chromadb() -> None:
    """Create the dedicated Chroma venv (if needed) and install chromadb into it.

    Deliberately isolated from berkie_reachy's own environment - see module
    docstring for why.
    """
    if not CHROMA_VENV_DIR.exists():
        logger.info("Creating isolated venv for ChromaDB at %s", CHROMA_VENV_DIR)
        subprocess.run([sys.executable, "-m", "venv", str(CHROMA_VENV_DIR)], check=True)

    logger.info("Installing chromadb into isolated venv...")
    subprocess.run(
        [str(_venv_python()), "-m", "pip", "install", "--quiet", "chromadb>=1.5"],
        check=True,
    )


def is_chroma_running() -> bool:
    """Return True if a local Chroma server is currently accepting connections."""
    return state.is_port_listening("127.0.0.1", state.CHROMA_PORT)


def ensure_chroma_running(*, timeout: float = 30.0) -> Optional[int]:
    """Start local ChromaDB (from its isolated venv) if not already running.

    Returns the pid this call started, or None if reused. Raises ImportError
    if chromadb isn't installed yet in the isolated venv (caller should offer
    install_chromadb()).
    """
    if is_chroma_running():
        logger.info("Reusing already-running Chroma on port %s", state.CHROMA_PORT)
        return None

    if not is_chromadb_installed():
        raise ImportError("chromadb is not installed in the isolated venv yet")

    state.ensure_dirs()
    logfile = state.LOGS_DIR / "chroma.log"
    # --host is explicit: it defaults to "localhost", which on some machines
    # (verified on this one) resolves to IPv6 ::1 only - not 127.0.0.1, which
    # is what our health check and llm_engine's CHROMA_DB_URL both use.
    cmd = [
        str(_venv_chroma()),
        "run",
        "--path",
        str(state.CHROMA_DATA_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(state.CHROMA_PORT),
    ]
    with open(logfile, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_chroma_running():
            state.write_pidfile(PROCESS_NAME, proc.pid)
            logger.info("Started Chroma (pid %s) on port %s", proc.pid, state.CHROMA_PORT)
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"chroma exited early (code {proc.returncode}); see {logfile}")
        time.sleep(0.3)

    proc.terminate()
    raise TimeoutError(f"Chroma did not start listening on port {state.CHROMA_PORT} within {timeout}s")


def stop_chroma_if_started_by_us() -> None:
    """Stop the Chroma process this run started, if any."""
    pid = state.read_pidfile(PROCESS_NAME)
    if pid is None:
        return
    logger.info("Stopping Chroma (pid %s)", pid)
    state.stop_pid(pid)
    state.clear_pidfile(PROCESS_NAME)
