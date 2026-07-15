"""Persistent local state for the bundled llm_engine backend.

Everything here lives under ``~/.berkie_reachy/llm_backend`` rather than the
Reachy Mini app's own ``instance_path``, because ``instance_path`` can be wiped
out and recreated when the app is reinstalled/updated, while the cloned
llm_engine source and its Mongo/Chroma data directories should survive that.
"""

from __future__ import annotations

import os
import shutil
import socket
import signal
import secrets
import logging
import contextlib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reachy Mini apps are launched by the daemon (reachy-mini-daemon), which the
# app-runner (reachy_mini/apps/manager.py) spawns the app subprocess from by
# copying the daemon's *own* environment (os.environ.copy(), not a fresh
# minimal one). So this isn't about the app subprocess getting a stripped
# PATH - it's that the daemon *itself* is commonly launched via a GUI app
# (e.g. a menu-bar/desktop launcher) rather than an interactive shell, and
# GUI-launched processes on macOS/Linux typically get a minimal default PATH
# (e.g. "/usr/bin:/bin:/usr/sbin:/sbin") that doesn't include Homebrew's
# /opt/homebrew/bin or /usr/local/bin - so shutil.which() alone reports tools
# as "not found" even when they're genuinely installed. Check common install
# locations directly as a fallback, rather than only trusting PATH.
_EXTRA_BIN_DIRS = [
    "/opt/homebrew/bin",  # Homebrew, Apple Silicon
    "/opt/homebrew/sbin",
    "/usr/local/bin",  # Homebrew, Intel Mac; also common on Linux
    "/usr/local/sbin",
    "/usr/local/opt/node/bin",
    "/usr/local/opt/mongodb-community/bin",
]


def find_executable(name: str) -> Optional[str]:
    """Locate an executable by name, checking PATH first and then common install dirs.

    Plain shutil.which() alone misses this on a GUI-launched daemon (see
    module docstring above for why); a real install shouldn't be reported
    as "not found" just because of how the daemon process happened to start.
    """
    found = shutil.which(name)
    if found:
        return found
    for d in _EXTRA_BIN_DIRS:
        candidate = Path(d) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None

APP_DATA_ROOT = Path.home() / ".berkie_reachy" / "llm_backend"

LLM_ENGINE_SRC_DIR = APP_DATA_ROOT / "llm_engine-src"
MONGO_DATA_DIR = APP_DATA_ROOT / "mongo-data"
CHROMA_DATA_DIR = APP_DATA_ROOT / "chroma-data"
RUN_DIR = APP_DATA_ROOT / "run"
LOGS_DIR = APP_DATA_ROOT / "logs"
RUNTIME_ENV_PATH = APP_DATA_ROOT / "runtime.env"

MONGO_PORT = 27017
CHROMA_PORT = 8002  # NOT 8000 - the reachy_mini daemon-detection probe uses 8000
LLM_ENGINE_PORT = 3000
LLM_ENGINE_WS_PORT = 5555


def ensure_dirs() -> None:
    """Create all persistent state directories if missing."""
    for d in (APP_DATA_ROOT, LLM_ENGINE_SRC_DIR.parent, MONGO_DATA_DIR, CHROMA_DATA_DIR, RUN_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if something is accepting TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pidfile_path(name: str) -> Path:
    return RUN_DIR / f"{name}.pid"


def read_pidfile(name: str) -> Optional[int]:
    """Read a previously written pid for ``name``, or None if absent/invalid."""
    p = _pidfile_path(name)
    try:
        text = p.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (FileNotFoundError, ValueError):
        return None


def write_pidfile(name: str, pid: int) -> None:
    """Persist a process's pid so a later run can detect/reuse/stop it."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _pidfile_path(name).write_text(str(pid), encoding="utf-8")


def clear_pidfile(name: str) -> None:
    """Remove a pidfile, if present."""
    with contextlib.suppress(FileNotFoundError):
        _pidfile_path(name).unlink()


def pid_is_alive(pid: int) -> bool:
    """Return True if a process with this pid currently exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else - still "alive" for our purposes.
        return True
    return True


def stop_pid(pid: int, *, term_signal: int = signal.SIGTERM, timeout: float = 10.0) -> None:
    """Ask a process to stop, escalating to SIGKILL if it doesn't within timeout."""
    import time

    if not pid_is_alive(pid):
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, term_signal)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def get_or_create_jwt_secret() -> str:
    """Load a persisted JWT secret for the local llm_engine instance, generating one on first use.

    Keeping this stable across restarts means the Berky operator account's
    login tokens (and the bootstrap's own auth) don't need to be re-minted
    every single launch.
    """
    ensure_dirs()
    key = "LLM_BACKEND_JWT_SECRET"
    if RUNTIME_ENV_PATH.exists():
        for line in RUNTIME_ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    secret = secrets.token_urlsafe(48)
    with RUNTIME_ENV_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{key}={secret}\n")
    return secret


class BootstrapLock:
    """Simple exclusive lockfile so two bootstrap runs never race to provision the same stack."""

    def __init__(self, name: str = "bootstrap"):
        self._path = RUN_DIR / f"{name}.lock"
        self._fd: Optional[int] = None

    def __enter__(self) -> "BootstrapLock":
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
        except FileExistsError:
            # Stale lock from a crashed previous run? Only steal it if that pid is dead.
            try:
                existing_pid = int(self._path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                existing_pid = None
            if existing_pid is not None and pid_is_alive(existing_pid):
                raise RuntimeError(
                    f"Another llm_engine_bootstrap run (pid {existing_pid}) already holds the lock."
                )
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()
            self._fd = os.open(str(self._path), os.O_CREAT | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
