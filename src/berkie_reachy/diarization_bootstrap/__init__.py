"""Provisions an isolated speaker-diarization backend for Berky.

pyannote.audio is installed into its own dedicated virtual environment
(``~/.berkie_reachy/diarization/venv``), NOT into berkie_reachy's own Python
environment - installing it into the shared environment was tried first and
rejected: it bumps ``protobuf`` to a version incompatible with ``mediapipe``
(used for optional head tracking), breaking `import mediapipe` outright
(confirmed empirically). Since diarization only needs to run as a background
service anyway (the ASR audio + segments are handed over, aligned speaker
labels come back), it's isolated behind a small local HTTP server
(``diarization_server.py``) run entirely inside that venv - berkie_reachy's
own process never imports pyannote/torch at all.

Diarization is optional and non-fatal by design: if installation or startup
fails for any reason, ``ensure_diarization_stack`` logs and returns None
rather than raising, and callers (``local_whisper.py``) already fall back to
plain (non-diarized) Whisper transcripts when that happens.
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

from berkie_reachy.llm_engine_bootstrap import state as llm_state

logger = logging.getLogger(__name__)

DIARIZATION_ROOT = Path.home() / ".berkie_reachy" / "diarization"
VENV_DIR = DIARIZATION_ROOT / "venv"
LOGS_DIR = DIARIZATION_ROOT / "logs"
RUN_DIR = DIARIZATION_ROOT / "run"

PORT = 8765  # distinct from mongo (27017), chroma (8002), llm_engine (3000/5555), daemon probe (8000)
BASE_URL = f"http://127.0.0.1:{PORT}"
PROCESS_NAME = "diarization_server"

_SERVER_SCRIPT = Path(__file__).parent / "diarization_server.py"

# Same PYTHONPATH/PYTHONHOME leak this project's gstreamer_env.py causes for
# every subprocess (see llm_engine_bootstrap/chroma.py for the full story) -
# strip it here too so pip/python actually see this venv's own site-packages.
_LEAKING_ENV_VARS = ("PYTHONPATH", "PYTHONHOME")


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in _LEAKING_ENV_VARS:
        env.pop(key, None)
    return env


def _ensure_dirs() -> None:
    for d in (DIARIZATION_ROOT, LOGS_DIR, RUN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _venv_python() -> Path:
    return VENV_DIR / "bin" / "python3"


def is_installed() -> bool:
    """Return True if pyannote.audio is importable in the isolated venv."""
    if not _venv_python().exists():
        return False
    result = subprocess.run(
        [str(_venv_python()), "-c", "import pyannote.audio"],
        capture_output=True,
        env=_clean_env(),
    )
    return result.returncode == 0


def _import_error() -> str:
    if not _venv_python().exists():
        return f"venv python not found at {_venv_python()}"
    result = subprocess.run(
        [str(_venv_python()), "-c", "import pyannote.audio"],
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    return (result.stderr or result.stdout or "(no output)").strip()[-2000:]


def install() -> None:
    """Create the isolated venv (if needed) and install pyannote.audio into it.

    A fresh install pulls in torch + pytorch-lightning + pyannote's own stack -
    easily several hundred MB - so this uses a generous timeout. Verifies the
    import actually succeeds afterward (pip reporting success doesn't always
    mean the package is importable - see chroma.py's install for the same
    lesson learned the hard way); deletes the venv on failure so the next
    attempt starts clean.
    """
    _ensure_dirs()
    clean_env = _clean_env()

    if not VENV_DIR.exists():
        logger.info("Creating isolated venv for diarization at %s", VENV_DIR)
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True, env=clean_env)

    logger.info("Installing pyannote.audio into isolated venv (can take several minutes on first run)...")
    pip_result = subprocess.run(
        [str(_venv_python()), "-m", "pip", "install", "pyannote.audio>=3.3.0", "soundfile", "sounddevice"],
        capture_output=True,
        text=True,
        timeout=1200,
        env=clean_env,
    )
    if pip_result.returncode != 0:
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        raise RuntimeError(f"pip install failed (exit {pip_result.returncode}):\n{(pip_result.stderr or pip_result.stdout).strip()[-2000:]}")

    if not is_installed():
        import_error = _import_error()
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        raise RuntimeError(
            "pip install reported success but pyannote.audio still isn't importable in the "
            f"isolated venv. Import error:\n{import_error}\nDeleted the venv so the next attempt starts fresh."
        )


def is_server_healthy() -> bool:
    """Return True if the diarization server is up and responding."""
    try:
        import httpx

        resp = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_server_running(*, hf_token: Optional[str], device: str = "cpu", timeout: float = 30.0) -> Optional[int]:
    """Start the diarization server (from its isolated venv) if not already running.

    Returns the pid this call started, or None if reused. Raises if pyannote
    isn't installed yet - caller should install() first.
    """
    if is_server_healthy():
        logger.info("Reusing already-running diarization server on port %s", PORT)
        return None

    if not is_installed():
        raise ImportError("pyannote.audio is not installed in the isolated diarization venv yet")

    _ensure_dirs()
    logfile = LOGS_DIR / "diarization_server.log"
    cmd = [str(_venv_python()), str(_SERVER_SCRIPT), "--port", str(PORT), "--device", device]
    if hf_token:
        cmd += ["--hf-token", hf_token]

    with open(logfile, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=_clean_env())

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_healthy():
            llm_state.write_pidfile(PROCESS_NAME, proc.pid)
            logger.info("Started diarization server (pid %s) on port %s", proc.pid, PORT)
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"diarization server exited early (code {proc.returncode}); see {logfile}")
        time.sleep(0.3)

    proc.terminate()
    raise TimeoutError(f"diarization server did not start listening on port {PORT} within {timeout}s")


def stop_server_if_started_by_us() -> None:
    """Stop the diarization server process this run started, if any."""
    pid = llm_state.read_pidfile(PROCESS_NAME)
    if pid is None:
        return
    logger.info("Stopping diarization server (pid %s)", pid)
    llm_state.stop_pid(pid)
    llm_state.clear_pidfile(PROCESS_NAME)


_cached_base_url: Optional[str] = None
_attempted = False


def ensure_diarization_stack(*, hf_token: Optional[str], device: str = "cpu") -> Optional[str]:
    """Ensure the isolated diarization venv+server are ready; return its base URL, or None.

    Memoized per-process: the actual install/health-check work only happens
    once - later calls (e.g. a fresh LocalWhisperSegmenter per stream session)
    just return the cached result immediately. Never raises - diarization is
    an optional enhancement, so any failure here just means callers fall back
    to plain (non-diarized) transcripts, matching local_whisper.py's existing
    non-fatal handling.
    """
    global _cached_base_url, _attempted
    if _attempted:
        return _cached_base_url
    _attempted = True

    try:
        if not is_installed():
            install()
        ensure_server_running(hf_token=hf_token, device=device)
        _cached_base_url = BASE_URL
    except Exception:
        logger.exception("Diarization backend unavailable; falling back to plain Whisper transcripts")
        _cached_base_url = None

    return _cached_base_url
