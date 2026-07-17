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

import os
import sys
import shutil
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

# This project's own Python framework build injects PYTHONPATH/PYTHONHOME
# (via gstreamer_env.py's GStreamer bundle setup) pointing at the *system*
# framework site-packages into every process it starts. Left inherited, that
# leak defeats venv isolation: pip sees system-installed packages as "already
# satisfied" and skips installing them into the venv, so the venv ends up
# missing real dependencies (caught in practice: chromadb installed and
# "verified" successfully here, but only because the verification subprocess
# inherited the same leaked PYTHONPATH the install did - a clean invocation,
# e.g. from a plain shell or a host without this leak, correctly showed
# `import chromadb` failing with `ModuleNotFoundError: No module named
# 'typing_extensions'`). Strip these for every subprocess that touches the
# Chroma venv so it's actually self-contained regardless of what the calling
# process's environment looks like.
_LEAKING_ENV_VARS = ("PYTHONPATH", "PYTHONHOME")


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in _LEAKING_ENV_VARS:
        env.pop(key, None)
    return env


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
        env=_clean_env(),
    )
    return result.returncode == 0


def _import_chromadb_error() -> str:
    """Run the same import check as is_chromadb_installed() but return the actual error text."""
    if not _venv_python().exists():
        return f"venv python not found at {_venv_python()}"
    result = subprocess.run(
        [str(_venv_python()), "-c", "import chromadb"],
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    return (result.stderr or result.stdout or "(no output)").strip()[-2000:]


def _free_disk_space_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return -1.0


def install_chromadb() -> None:
    """Create the dedicated Chroma venv (if needed) and install chromadb into it.

    Deliberately isolated from berkie_reachy's own environment - see module
    docstring for why. A fresh (uncached) install can take a couple of
    minutes - chromadb's own dependency tree is large (onnxruntime, grpcio,
    etc., easily several hundred MB) - so this uses a generous timeout and
    explicitly re-checks success afterward rather than only trusting that
    pip's exit code was 0 (seen in practice: pip reporting success while the
    package still isn't importable, e.g. from a disk-space-exhausted install
    or a stale/partially-corrupted venv left by an earlier interrupted run).

    On failure, the venv is deleted so the next attempt starts clean rather
    than repeatedly retrying against a possibly-broken install.
    """
    state.ensure_dirs()
    free_gb = _free_disk_space_gb(state.APP_DATA_ROOT)
    clean_env = _clean_env()

    if not CHROMA_VENV_DIR.exists():
        logger.info("Creating isolated venv for ChromaDB at %s (%.1f GB free)", CHROMA_VENV_DIR, free_gb)
        subprocess.run([sys.executable, "-m", "venv", str(CHROMA_VENV_DIR)], check=True, env=clean_env)

    logger.info("Installing chromadb into isolated venv (can take a few minutes on first run)...")
    pip_result = subprocess.run(
        [str(_venv_python()), "-m", "pip", "install", "chromadb>=1.5"],
        capture_output=True,
        text=True,
        timeout=600,
        env=clean_env,
    )
    if pip_result.returncode != 0:
        shutil.rmtree(CHROMA_VENV_DIR, ignore_errors=True)
        raise RuntimeError(
            f"pip install failed (exit {pip_result.returncode}, {free_gb:.1f} GB free at start):\n"
            f"{(pip_result.stderr or pip_result.stdout).strip()[-2000:]}"
        )

    if not is_chromadb_installed():
        import_error = _import_chromadb_error()
        shutil.rmtree(CHROMA_VENV_DIR, ignore_errors=True)
        raise RuntimeError(
            "pip install reported success but chromadb still isn't importable in the "
            f"isolated venv ({free_gb:.1f} GB free at install start). Import error:\n{import_error}\n"
            "Deleted the venv so the next attempt starts from scratch."
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
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=_clean_env())

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
