"""Browser-facing Berky live handler.

This handler is used by the Reachy Mini Gradio app on port 7860.
It accepts browser microphone audio, transcribes it locally with Whisper,
sends finalized transcript chunks to LLM Engine over Socket.IO, and
synthesizes agent replies to raw audio samples for playback through the
robot's own speaker.
"""

from __future__ import annotations

import base64
import asyncio
import logging
from typing import Any, Optional, Tuple

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from numpy.typing import NDArray

from berkie_reachy.llm_engine_socket import LLMEngineSocketClient, _message_text
from berkie_reachy.local_whisper import LocalWhisperSegmenter
from berkie_reachy.tts import CommandTTS
from berkie_reachy.audio.head_wobbler import SAMPLE_RATE as WOBBLER_SAMPLE_RATE


logger = logging.getLogger(__name__)


class BerkyLiveHandler(AsyncStreamHandler):
    """Stream browser audio to Berky via local Whisper and LLM Engine."""

    def __init__(self, movement_manager: Optional[Any] = None, head_wobbler: Optional[Any] = None) -> None:
        super().__init__(expected_layout="mono", output_sample_rate=16000, input_sample_rate=16000)
        self.output_queue: "asyncio.Queue[AdditionalOutputs | Tuple[int, NDArray[Any]]]" = asyncio.Queue()
        self.transcriber = LocalWhisperSegmenter()
        self.tts = CommandTTS()
        self.client = LLMEngineSocketClient(on_agent_message=self._on_agent_message)
        self._connected = False
        self._movement_manager = movement_manager
        self._head_wobbler = head_wobbler

    def copy(self) -> "BerkyLiveHandler":
        """Create a fresh handler for a new stream session."""
        return BerkyLiveHandler(movement_manager=self._movement_manager, head_wobbler=self._head_wobbler)

    async def start_up(self) -> None:
        """Connect to LLM Engine before audio starts flowing."""
        if self._connected:
            return
        await self.client.connect()
        self._connected = True
        logger.info("Berky live handler connected to LLM Engine")

    def _feed_head_wobbler(self, samples: NDArray[np.int16], sample_rate: int) -> None:
        """Drive audio-cadence head movement from the synthesized speech itself.

        HeadWobbler/SwayRollRT analyze the actual audio envelope (loudness,
        voice-activity attack/release) to sway the head in step with real
        speech rhythm, rather than a generic fixed animation - it just always
        assumes SAMPLE_RATE-rate PCM16 input (matching the OpenAI realtime
        API's output format, its other caller), so resample to that first.
        """
        if self._head_wobbler is None:
            return
        audio = samples
        if sample_rate != WOBBLER_SAMPLE_RATE:
            audio = resample(audio.astype(np.float32), int(len(audio) * WOBBLER_SAMPLE_RATE / sample_rate))
            audio = audio.astype(np.int16)
        self._head_wobbler.reset()
        self._head_wobbler.feed(base64.b64encode(audio.tobytes()).decode("ascii"))

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

        if synth is not None:
            sample_rate, samples = synth
            self._feed_head_wobbler(samples, sample_rate)
            await self.output_queue.put(synth)
            await asyncio.sleep(len(samples) / sample_rate)
            if self._movement_manager is not None:
                self._movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            # No file-capable TTS binary found; fall back to direct playback
            # on this machine so the response is at least audible somewhere.
            await self.tts.speak(text)

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
