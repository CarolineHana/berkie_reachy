"""Reachy-side Berky runtime.

This process is the physical layer:
- reads microphone audio from Reachy,
- transcribes local speech with faster-whisper,
- sends finalized transcript chunks to LLM Engine over Socket.IO,
- speaks Berky agent responses as they arrive.
"""

from __future__ import annotations

import time
import os
import importlib.util
import asyncio
import logging
import argparse
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from reachy_mini import ReachyMini

from berkie_reachy.tts import CommandTTS
from berkie_reachy.utils import setup_logger
from berkie_reachy.local_whisper import LocalWhisperSegmenter
from berkie_reachy.llm_engine_socket import LLMEngineSocketClient, _message_text


logger = logging.getLogger(__name__)


def _prepend_env_path(name: str, values: list[Path]) -> None:
    existing = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    new_values = [str(value) for value in values if value.exists()]
    merged = []
    for item in [*new_values, *existing]:
        if item not in merged:
            merged.append(item)
    if merged:
        os.environ[name] = os.pathsep.join(merged)


def configure_gstreamer_bundle_env() -> None:
    """Prefer pip's bundled GStreamer libraries over older conda libraries.

    The Reachy Mini daemon imports GStreamer unconditionally. In mixed Anaconda
    environments, conda's older libgstreamer can be chosen before the pip
    bundle and causes missing-symbol failures. These env vars are inherited by
    the daemon process spawned by the SDK.
    """
    spec = importlib.util.find_spec("gstreamer_libs")
    if spec is None or spec.origin is None:
        return

    site_packages = Path(spec.origin).resolve().parent.parent
    lib_dir = site_packages / "gstreamer_libs" / "lib"
    python_lib_dir = Path(os.__file__).resolve().parents[1]
    plugin_dirs = [
        site_packages / package / "lib" / "gstreamer-1.0"
        for package in (
            "gstreamer_libs",
            "gstreamer_plugins",
            "gstreamer_plugins_libs",
            "gstreamer_plugins_gpl",
            "gstreamer_plugins_restricted",
            "gstreamer_plugins_gpl_restricted",
            "gstreamer_gtk",
            "gstreamer_python",
        )
    ]
    typelib_dirs = [
        site_packages / "gstreamer_libs" / "lib" / "girepository-1.0",
        site_packages / "gstreamer_python" / "lib" / "girepository-1.0",
        site_packages / "gstreamer_gtk" / "lib" / "girepository-1.0",
    ]

    _prepend_env_path("DYLD_LIBRARY_PATH", [lib_dir])
    _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", [python_lib_dir])
    _prepend_env_path("GST_PLUGIN_SYSTEM_PATH_1_0", plugin_dirs)
    _prepend_env_path("GI_TYPELIB_PATH", typelib_dirs)


def parse_args() -> argparse.Namespace:
    """Parse Berky runtime arguments."""
    parser = argparse.ArgumentParser(description="Run Berky on Reachy with LLM Engine live transcript streaming.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--robot-name", default=None, help="Optional Reachy robot name.")
    parser.add_argument(
        "--virtual-reachy",
        action="store_true",
        help="Spawn/connect to the MuJoCo simulated Reachy Mini daemon.",
    )
    parser.add_argument(
        "--mockup-sim",
        action="store_true",
        help="Spawn/connect to the lightweight Reachy mockup daemon for local testing.",
    )
    parser.add_argument(
        "--spawn-daemon",
        action="store_true",
        help="Ask the Reachy SDK to spawn a daemon before connecting.",
    )
    parser.add_argument("--host", default="reachy-mini.local", help="Reachy daemon host.")
    parser.add_argument("--port", type=int, default=8000, help="Reachy daemon FastAPI port.")
    parser.add_argument(
        "--connection-mode",
        choices=["auto", "localhost_only", "network"],
        default="auto",
        help="Reachy SDK connection mode.",
    )
    parser.add_argument(
        "--media-backend",
        default="default",
        help='Reachy media backend. Use "no_media" for simulator smoke tests without audio.',
    )
    parser.add_argument(
        "--input-mode",
        choices=["robot_audio", "stdin"],
        default="robot_audio",
        help="Transcript input source. Use stdin for virtual no-media testing.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Reachy connection timeout in seconds.")
    parser.add_argument(
        "--daemon-startup-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for a mockup daemon started by this process.",
    )
    parser.add_argument(
        "--robot-smoke-test",
        action="store_true",
        help="Connect to Reachy, print daemon status, then exit before starting Whisper or LLM Engine.",
    )
    args = parser.parse_args()
    if args.virtual_reachy and args.mockup_sim:
        parser.error("Use either --virtual-reachy for MuJoCo or --mockup-sim for the lightweight daemon, not both.")
    if args.mockup_sim:
        args.host = "localhost"
        args.connection_mode = "localhost_only"
        if args.robot_smoke_test and args.media_backend == "default":
            args.media_backend = "no_media"
    return args


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Reachy mockup daemon exited with status {process.returncode}.")
        if _is_port_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Reachy mockup daemon did not listen on {host}:{port} within {timeout:.1f}s.")


def _maybe_start_mockup_daemon(args: argparse.Namespace) -> subprocess.Popen[Any] | None:
    if not args.mockup_sim:
        return None

    host = "127.0.0.1"
    if _is_port_open(host, args.port):
        logger.info("Using existing Reachy daemon on %s:%s", host, args.port)
        return None

    cmd = [
        "reachy-mini-daemon",
        "--mockup-sim",
        "--fastapi-host",
        host,
        "--fastapi-port",
        str(args.port),
    ]
    if args.media_backend == "no_media":
        cmd.append("--no-media")

    logger.info("Starting Reachy mockup daemon: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, start_new_session=True)
    _wait_for_port(host, args.port, args.daemon_startup_timeout, process)
    return process


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    logger.info("Stopping Reachy mockup daemon")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class BerkyReachyRuntime:
    """Owns the live robot/audio/socket lifecycle."""

    def __init__(self, robot: ReachyMini, *, input_mode: str = "robot_audio") -> None:
        self.robot = robot
        self.input_mode = input_mode
        self.tts = CommandTTS()
        self.transcriber = LocalWhisperSegmenter()
        self.stop_event = asyncio.Event()
        self._movement_manager: Any | None = None
        self._movement_thread_started = False
        self.client = LLMEngineSocketClient(on_agent_message=self._on_agent_message)

    async def _on_agent_message(self, message: dict[str, Any]) -> None:
        text = _message_text(message)
        if not text:
            return
        logger.info("Berky agent response: %s", text)
        await self.tts.speak(text)

    def _start_motion(self) -> None:
        try:
            from berkie_reachy.moves import MovementManager

            self._movement_manager = MovementManager(current_robot=self.robot, camera_worker=None)
            self._movement_manager.start()
            self._movement_thread_started = True
        except Exception as exc:
            logger.warning("Movement manager unavailable; continuing without expression motion: %s", exc)

    def _set_listening(self, listening: bool) -> None:
        if self._movement_manager is None:
            return
        try:
            self._movement_manager.set_listening(listening)
        except Exception:
            logger.debug("Failed to update listening state", exc_info=True)

    async def run(self) -> None:
        """Run until interrupted."""
        self._start_motion()
        await self.client.connect()
        if self.input_mode == "stdin":
            await self._run_stdin_transcripts()
            return

        input_sample_rate = self.robot.media.get_input_audio_samplerate()
        self.robot.media.start_recording()
        logger.info("Reachy microphone recording started at %s Hz", input_sample_rate)

        try:
            while not self.stop_event.is_set():
                frame = self.robot.media.get_audio_sample()
                if frame is None:
                    await asyncio.sleep(0)
                    continue

                transcript = await self.transcriber.accept(input_sample_rate, frame)
                self._set_listening(self.transcriber.is_active)
                if transcript:
                    await self.client.send_transcript(transcript, final=True)

                await asyncio.sleep(0)
        finally:
            await self.shutdown()

    async def _run_stdin_transcripts(self) -> None:
        """Send each stdin line as a finalized transcript chunk."""
        logger.info("Reading transcript lines from stdin. Type /quit to stop.")
        loop = asyncio.get_running_loop()
        try:
            while not self.stop_event.is_set():
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if line == "":
                    await asyncio.sleep(0.2)
                    continue
                text = line.strip()
                if not text:
                    continue
                if text in {"/quit", "/exit"}:
                    self.stop_event.set()
                    break
                await self.client.send_transcript(text, final=True)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop media, socket, and motion resources."""
        logger.info("Shutting down Berky Reachy runtime")
        try:
            self.robot.media.stop_recording()
        except Exception:
            logger.debug("Error stopping recording", exc_info=True)

        await self.client.disconnect()

        if self._movement_manager is not None and self._movement_thread_started:
            try:
                self._movement_manager.stop()
            except Exception:
                logger.debug("Error stopping movement manager", exc_info=True)

        try:
            self.robot.media.close()
        except Exception:
            logger.debug("Error closing media", exc_info=True)

        try:
            self.robot.client.disconnect()
        except Exception:
            logger.debug("Error disconnecting robot client", exc_info=True)


def _build_robot(args: argparse.Namespace) -> ReachyMini:
    kwargs: dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "connection_mode": args.connection_mode,
        "spawn_daemon": bool(args.spawn_daemon or args.virtual_reachy) and not args.mockup_sim,
        "use_sim": bool(args.virtual_reachy) and not args.mockup_sim,
        "timeout": args.timeout,
        "media_backend": args.media_backend,
    }
    if args.robot_name:
        kwargs["robot_name"] = args.robot_name
    return ReachyMini(**kwargs)


def main() -> None:
    """CLI entry point for the Berky Reachy runtime."""
    args = parse_args()
    setup_logger(args.debug)
    configure_gstreamer_bundle_env()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    daemon_process = None

    try:
        daemon_process = _maybe_start_mockup_daemon(args)
        robot = _build_robot(args)
        if args.robot_smoke_test:
            print(robot.client.get_status())
            robot.client.disconnect()
            return

        runtime = BerkyReachyRuntime(robot, input_mode=args.input_mode)
        loop.run_until_complete(runtime.run())
    except KeyboardInterrupt:
        if "runtime" in locals():
            runtime.stop_event.set()
            loop.run_until_complete(runtime.shutdown())
    finally:
        time.sleep(0.2)
        loop.close()
        _stop_process(daemon_process)


if __name__ == "__main__":
    main()
