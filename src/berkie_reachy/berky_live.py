"""Browser-facing Berky live handler.

This handler is used by the Reachy Mini Gradio app on port 7860.
It accepts browser microphone audio, transcribes it locally with Whisper,
sends finalized transcript chunks to LLM Engine over Socket.IO, and
synthesizes agent replies to raw audio samples for playback through the
robot's own speaker.
"""

from __future__ import annotations

import math
import asyncio
import logging
from typing import Any, Optional, Tuple

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray

from berkie_reachy.llm_engine_socket import LLMEngineSocketClient, _message_text
from berkie_reachy.local_whisper import LocalWhisperSegmenter
from berkie_reachy.tts import CommandTTS


logger = logging.getLogger(__name__)

_SPEAKING_UPDATE_HZ = 20
_SPEAKING_PITCH_AMP = 0.1    # radians (~5.7°) visible nod
_SPEAKING_YAW_AMP = 0.05     # radians (~2.9°) subtle turn
_SPEAKING_PITCH_FREQ = 0.5   # Hz
_SPEAKING_YAW_FREQ = 0.3     # Hz


class BerkyLiveHandler(AsyncStreamHandler):
    """Stream browser audio to Berky via local Whisper and LLM Engine."""

    def __init__(self, movement_manager: Optional[Any] = None) -> None:
        super().__init__(expected_layout="mono", output_sample_rate=16000, input_sample_rate=16000)
        self.output_queue: "asyncio.Queue[AdditionalOutputs | Tuple[int, NDArray[Any]]]" = asyncio.Queue()
        self.transcriber = LocalWhisperSegmenter()
        self.tts = CommandTTS()
        self.client = LLMEngineSocketClient(on_agent_message=self._on_agent_message)
        self._connected = False
        self._movement_manager = movement_manager

    def copy(self) -> "BerkyLiveHandler":
        """Create a fresh handler for a new stream session."""
        return BerkyLiveHandler(movement_manager=self._movement_manager)

    async def start_up(self) -> None:
        """Connect to LLM Engine before audio starts flowing."""
        if self._connected:
            return
        await self.client.connect()
        self._connected = True
        logger.info("Berky live handler connected to LLM Engine")

    async def _speaking_animation(self, stop_event: asyncio.Event) -> None:
        """Apply gentle sinusoidal head movement while speaking."""
        dt = 1.0 / _SPEAKING_UPDATE_HZ
        t = 0.0
        try:
            while not stop_event.is_set():
                pitch = _SPEAKING_PITCH_AMP * math.sin(2 * math.pi * _SPEAKING_PITCH_FREQ * t)
                yaw = _SPEAKING_YAW_AMP * math.sin(2 * math.pi * _SPEAKING_YAW_FREQ * t + 1.0)
                self._movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, pitch, yaw))
                t += dt
                await asyncio.sleep(dt)
        finally:
            self._movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    async def _on_agent_message(self, message: dict[str, Any]) -> None:
        """Speak and display one agent message.

        Synthesizes to raw samples and enqueues them as a plain tuple, which
        console.py's play_loop recognizes and pushes through robot.media -
        the robot's own speaker - rather than calling self.tts.speak(), which
        would play through this host machine's local audio output instead.
        """
        text = _message_text(message)
        if not text:
            return

        synth = await self.tts.synthesize(text)
        duration = (len(synth[1]) / synth[0]) if synth is not None else 0.0

        if self._movement_manager is not None:
            stop_anim = asyncio.Event()
            anim_task = asyncio.create_task(self._speaking_animation(stop_anim))

        if synth is not None:
            await self.output_queue.put(synth)
        else:
            # No file-capable TTS binary found; fall back to direct playback
            # on this machine so the response is at least audible somewhere.
            await self.tts.speak(text)

        if self._movement_manager is not None:
            if duration > 0:
                await asyncio.sleep(duration)
            stop_anim.set()
            await anim_task

        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))

    async def receive(self, frame: Tuple[int, NDArray[Any]]) -> None:
        """Accept browser microphone frames and send completed transcripts."""
        if not self._connected:
            return

        sample_rate, audio = frame
        transcript = await self.transcriber.accept(sample_rate, audio)

        is_active = self.transcriber.is_active
        if self._movement_manager is not None:
            self._movement_manager.set_listening(is_active)

        if not transcript:
            return

        logger.info("Browser transcript: %s", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        try:
            await self.client.send_transcript(transcript, final=True)
        except Exception:
            logger.warning("Failed to send transcript — LLM Engine disconnected, will retry on reconnect")

    async def emit(self) -> AdditionalOutputs | Tuple[int, NDArray[Any]] | None:
        """Emit chatbot updates when they are available."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Disconnect from LLM Engine and clear pending output."""
        try:
            await self.client.disconnect()
        finally:
            self._connected = False
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
