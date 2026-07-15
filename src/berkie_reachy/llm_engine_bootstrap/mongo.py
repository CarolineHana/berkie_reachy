"""Detect/start/stop a local MongoDB instance for the bundled llm_engine backend."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from berkie_reachy.llm_engine_bootstrap import state

logger = logging.getLogger(__name__)

PROCESS_NAME = "mongod"
MONGODB_URL = f"mongodb://127.0.0.1:{state.MONGO_PORT}/llm_engine"


class MongoNotAvailableError(RuntimeError):
    """Raised when the mongod binary isn't installed on this host."""


def _mongod_path() -> Optional[str]:
    """Resolve mongod's path, checking common install dirs beyond just PATH.

    See state.find_executable's docstring: the daemon that launches this app
    is often started via a GUI launcher with a minimal PATH, so plain
    shutil.which() can miss a genuinely-installed mongod.
    """
    return state.find_executable("mongod")


def is_mongod_available() -> bool:
    """Return True if the ``mongod`` binary can be found."""
    return _mongod_path() is not None


def is_mongo_running() -> bool:
    """Return True if a local MongoDB is currently accepting connections."""
    return state.is_port_listening("127.0.0.1", state.MONGO_PORT)


def ensure_mongo_running(*, timeout: float = 30.0) -> Optional[int]:
    """Start local MongoDB if not already running. Returns the pid this call started, or None if reused.

    Raises MongoNotAvailableError if mongod isn't installed.
    """
    if is_mongo_running():
        logger.info("Reusing already-running MongoDB on port %s", state.MONGO_PORT)
        return None

    mongod_path = _mongod_path()
    if mongod_path is None:
        raise MongoNotAvailableError(
            "mongod not found. Install MongoDB (e.g. `brew install mongodb-community` "
            "on macOS, or your distro's mongodb-org package on Linux), then relaunch the app."
        )

    state.ensure_dirs()
    logfile = state.LOGS_DIR / "mongod.log"
    proc = subprocess.Popen(
        [
            mongod_path,
            "--dbpath",
            str(state.MONGO_DATA_DIR),
            "--port",
            str(state.MONGO_PORT),
            "--bind_ip",
            "127.0.0.1",
            "--logpath",
            str(logfile),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_mongo_running():
            state.write_pidfile(PROCESS_NAME, proc.pid)
            logger.info("Started MongoDB (pid %s) on port %s", proc.pid, state.MONGO_PORT)
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"mongod exited early (code {proc.returncode}); see {logfile}")
        time.sleep(0.3)

    proc.terminate()
    raise TimeoutError(f"MongoDB did not start listening on port {state.MONGO_PORT} within {timeout}s")


def stop_mongo_if_started_by_us() -> None:
    """Stop the MongoDB process this run started, if any."""
    pid = state.read_pidfile(PROCESS_NAME)
    if pid is None:
        return
    logger.info("Stopping MongoDB (pid %s)", pid)
    state.stop_pid(pid)
    state.clear_pidfile(PROCESS_NAME)
