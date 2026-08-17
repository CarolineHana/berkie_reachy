"""Reachy-side Berky runtime.

This process is the physical layer:
- reads microphone audio from Reachy,
- transcribes local speech with faster-whisper,
- sends finalized transcript chunks to LLM Engine over Socket.IO,
- speaks Berky agent responses as they arrive.
"""

from __future__ import annotations
import os
import sys
import time
import socket
import asyncio
import logging
import argparse
import subprocess
import importlib.util
from typing import Any, Callable
from pathlib import Path

from fastrtc import audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from berkie_reachy.tts import CommandTTS, split_into_speech_sentences
from berkie_reachy.moves import start_thinking_motion
from berkie_reachy.utils import setup_logger
from berkie_reachy.config import config
from berkie_reachy.local_whisper import LocalWhisperSegmenter
from berkie_reachy.openai_realtime import contains_wake_phrase
from berkie_reachy.llm_engine_socket import LLMEngineSocketClient, _message_text


logger = logging.getLogger(__name__)

# Longer than any observed real response (worst case seen ~18s for a tool-heavy archive
# question) - bounds how long the thinking motion runs if a wake-ish transcript never
# actually gets a response (e.g. a mis-transcription that doesn't match server-side either).
THINKING_TIMEOUT_SECONDS = 30.0


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
        self.client = LLMEngineSocketClient(
            on_agent_message=self._on_agent_message,
            on_answer_chunk=self._on_answer_chunk,
        )
        # Streaming (see llm_engine's llmChain.ts streamAgentAndReportChunks): sentences
        # arrive one at a time, well before the full response is ready, via
        # berky:answer_chunk. _chunk_queue/_chunk_worker_task speak them as they arrive;
        # _streamed_last_response flags to _on_agent_message that the full text it just
        # got has already been spoken, so it shouldn't be synthesized again.
        self._chunk_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._chunk_worker_task: asyncio.Task[None] | None = None
        self._streaming_active = False
        self._streamed_last_response = False
        # Head motion happens while Berkie is "thinking" (generating a response) and holds
        # still once actual audio starts - see moves.start_thinking_motion. _thinking_token
        # invalidates a stale watchdog if thinking starts again before an earlier one's
        # timeout fires (e.g. two quick wake attempts).
        self._thinking_stop: Callable[[], None] | None = None
        self._thinking_token = 0
        # requestId of the turn currently being spoken, or None between turns. Guards against
        # two turns' chunks landing in the same _chunk_queue and being spoken as one blended
        # answer (e.g. if the mic picks something up during the pre-first-chunk buffering
        # window and triggers a second turn before this one finishes) - see
        # llm_engine_socket.py's berky:answer_chunk handler for where request_id comes from.
        self._active_request_id: str | None = None

    def _begin_thinking(self) -> None:
        """Start head motion for a response that's presumably being generated.

        Called speculatively on any transcript that looks like a wake attempt, before we
        know whether the server will actually treat it as one - harmless if it doesn't,
        since the watchdog below stops the motion on its own if nothing ever answers.
        """
        self._end_thinking()
        self._thinking_token += 1
        token = self._thinking_token
        self._thinking_stop = start_thinking_motion(self._movement_manager)

        async def _watchdog() -> None:
            await asyncio.sleep(THINKING_TIMEOUT_SECONDS)
            if self._thinking_token == token:
                self._end_thinking()

        asyncio.create_task(_watchdog())

    def _end_thinking(self) -> None:
        """Stop head motion - called right before real audio starts, or by the watchdog."""
        if self._thinking_stop is not None:
            self._thinking_stop()
            self._thinking_stop = None

    async def _play_through_robot(self, text: str) -> None:
        """Synthesize ``text`` and play it through Reachy's own speaker.

        Mirrors console.py's play_loop / berky_live.py's synthesize()+push pattern:
        self.tts.speak()/speak_chunked() shell out to a local TTS binary that plays on
        *this* machine's audio output, which isn't what's wanted when the robot is
        physically present and should be heard responding through its own speaker.
        Falls back to local playback only if no file-capable TTS binary is available.
        """
        synth = await self.tts.synthesize(text)
        if synth is None:
            await self.tts.speak(text)
            return

        sample_rate, samples = synth
        audio_frame = audio_to_float32(samples)
        output_sample_rate = self.robot.media.get_output_audio_samplerate()
        if sample_rate != output_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * output_sample_rate / sample_rate))
            sample_rate = output_sample_rate

        self.robot.media.push_audio_sample(audio_frame)
        await asyncio.sleep(len(audio_frame) / sample_rate)

    async def _chunk_worker(self) -> None:
        """Background consumer: speak sentences off _chunk_queue one at a time, in order."""
        while True:
            sentence = await self._chunk_queue.get()
            try:
                await self._play_through_robot(sentence)
            except Exception:
                logger.warning("Failed to speak streamed chunk %r", sentence, exc_info=True)
            finally:
                self._chunk_queue.task_done()

    async def _on_answer_chunk(self, request_id: str, text: str, done: bool) -> None:
        """Speak one incremental sentence of a still-generating answer as it arrives.

        See llm_engine's llmChain.ts streamAgentAndReportChunks - this is the client side
        of that: instead of waiting for the full response (_on_agent_message), start
        speaking each sentence the moment it's generated.
        """
        if self._active_request_id is not None and request_id != self._active_request_id:
            logger.warning(
                "Dropping answer chunk for request_id=%s; a different turn (%s) is still active",
                request_id,
                self._active_request_id,
            )
            return

        if self._active_request_id is None:
            self._end_thinking()  # real audio is about to start - hold the head still for it
            self._active_request_id = request_id
            self._streaming_active = True
            if self._chunk_worker_task is None or self._chunk_worker_task.done():
                self._chunk_worker_task = asyncio.create_task(self._chunk_worker(), name="chunk-speaker")

        if text.strip():
            await self._chunk_queue.put(text.strip())

        if done:
            # Set this before awaiting playback below: the server emits the persisted final
            # message (-> _on_agent_message) right after this done signal, which can easily
            # arrive and get processed while the last queued sentences are still playing. If
            # _streamed_last_response isn't already True by then, _on_agent_message falls
            # through to its own full-text synthesis and speaks the whole answer a second time
            # on top of the tail end of the streamed version - confirmed live ("says the same
            # thing twice").
            self._streamed_last_response = True
            self._active_request_id = None
            await self._chunk_queue.join()
            self._streaming_active = False

    async def _on_agent_message(self, message: dict[str, Any]) -> None:
        """Speak one agent message.

        If this response already streamed sentence-by-sentence via
        _on_answer_chunk, it's already been fully spoken - nothing left to do.
        Otherwise (streaming didn't fire, e.g. an older agent or an error),
        fall back to synthesizing the whole text here, still sentence-by-
        sentence so playback starts on the first sentence rather than waiting
        for the entire response.
        """
        text = _message_text(message)
        if not text:
            return
        logger.info("Berky agent response: %s", text)

        if self._streamed_last_response:
            self._streamed_last_response = False
            return

        self._end_thinking()  # real audio is about to start - hold the head still for it
        for chunk in split_into_speech_sentences(text):
            await self._play_through_robot(chunk)

    def _start_motion(self) -> None:
        try:
            from berkie_reachy.moves import MovementManager

            self._movement_manager = MovementManager(current_robot=self.robot, camera_worker=None)
            self._movement_manager.start()
            self._movement_thread_started = True
        except Exception as exc:
            logger.warning("Movement manager unavailable; continuing without expression motion: %s", exc)

    async def run(self) -> None:
        """Run until interrupted."""
        self._start_motion()
        self.robot.media.start_playing()
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
                if transcript:
                    if contains_wake_phrase(transcript, config.BERKY_WAKE_PHRASE or ""):
                        self._begin_thinking()
                    await self.client.send_transcript(
                        transcript,
                        final=True,
                        speaker=self.transcriber.last_speaker,
                    )

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
                if contains_wake_phrase(text, config.BERKY_WAKE_PHRASE or ""):
                    self._begin_thinking()
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

        try:
            self.robot.media.stop_playing()
        except Exception:
            logger.debug("Error stopping playback", exc_info=True)

        if self._chunk_worker_task is not None and not self._chunk_worker_task.done():
            self._chunk_worker_task.cancel()

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
