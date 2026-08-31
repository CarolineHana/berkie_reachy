"""Browser-facing Berky live handler.

This handler is used by the Reachy Mini Gradio app on port 7860.
It accepts browser microphone audio, transcribes it locally with Whisper,
sends finalized transcript chunks to LLM Engine over Socket.IO, and
synthesizes agent replies to raw audio samples for playback through the
robot's own speaker.
"""

from __future__ import annotations
import random
import asyncio
import logging
from typing import Any, Tuple, Callable, Optional

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray

from berkie_reachy.tts import CommandTTS, split_into_speech_sentences
from berkie_reachy.moves import nod_along_with_audio, start_thinking_motion
from berkie_reachy.config import config
from berkie_reachy.local_whisper import LocalWhisperSegmenter
from berkie_reachy.openai_realtime import contains_wake_phrase
from berkie_reachy.llm_engine_socket import LLMEngineSocketClient, _message_text


logger = logging.getLogger(__name__)

# Longer than any observed real response (worst case seen ~18s for a tool-heavy archive
# question) - bounds how long the thinking motion runs if a wake-ish transcript never
# actually gets a response (e.g. a mis-transcription that doesn't match server-side either).
THINKING_TIMEOUT_SECONDS = 30.0

# Spoken the moment a wake phrase is detected, in parallel with the thinking motion -
# synthesized locally via self.tts and never touches llm_engine, so it plays with none of
# the LLM round-trip latency the real answer is stuck waiting on.
THINKING_ACK_LINES = [
    "Heard you loud and clear, let me find the answer for you.",
    "Let me look that up for you.",
]


class BerkyLiveHandler(AsyncStreamHandler):
    """Stream browser audio to Berky via local Whisper and LLM Engine."""

    def __init__(self, movement_manager: Optional[Any] = None, interaction_mode: Optional[Any] = None) -> None:
        super().__init__(expected_layout="mono", output_sample_rate=16000, input_sample_rate=16000)
        self.output_queue: "asyncio.Queue[AdditionalOutputs | Tuple[int, NDArray[Any]]]" = asyncio.Queue()
        self.transcriber = LocalWhisperSegmenter()
        self.tts = CommandTTS()
        self.client = LLMEngineSocketClient(
            on_agent_message=self._on_agent_message,
            on_answer_chunk=self._on_answer_chunk,
        )
        self._connected = False
        self._movement_manager = movement_manager
        self._interaction_mode = interaction_mode
        self._speaking = False
        # Streaming (see llm_engine's llmChain.ts streamAgentAndReportChunks): sentences
        # arrive one at a time, well before the full response is ready, via
        # berky:answer_chunk. Two chained queues give this the same one-sentence-lookahead
        # overlap _on_agent_message's own fallback path gets: _synth_worker_task synthesizes
        # sentences off _chunk_queue continuously (independent of playback pacing) into
        # _synth_queue (maxsize=1, so it only ever runs one sentence ahead), while
        # _chunk_worker_task just plays whatever's ready in order - so synthesis of the next
        # sentence overlaps with playback of the current one instead of only starting once
        # playback finishes, which left an audible silent gap (TTS render time) between every
        # sentence and made Berkie sound like it had stopped talking mid-answer.
        # _streamed_last_response flags to _on_agent_message that the full text it just
        # got has already been spoken, so it shouldn't be synthesized again.
        self._chunk_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._synth_queue: "asyncio.Queue[Optional[Tuple[int, NDArray[Any]]]]" = asyncio.Queue(maxsize=1)
        self._synth_worker_task: Optional[asyncio.Task[None]] = None
        self._chunk_worker_task: Optional[asyncio.Task[None]] = None
        self._streaming_active = False
        self._streamed_last_response = False
        # Head motion happens while Berkie is "thinking" (generating a response) and holds
        # still once actual audio starts - see moves.start_thinking_motion. _thinking_token
        # invalidates a stale watchdog if thinking starts again before an earlier one's
        # timeout fires (e.g. two quick wake attempts).
        self._thinking_stop: Optional[Callable[[], None]] = None
        self._thinking_token = 0
        # requestId of the turn currently being spoken, or None between turns. Guards against
        # two turns' chunks landing in the same _chunk_queue and being spoken as one blended
        # answer (e.g. if the mic picks something up during the pre-first-chunk buffering
        # window and triggers a second turn before this one finishes) - see
        # llm_engine_socket.py's berky:answer_chunk handler for where request_id comes from.
        self._active_request_id: Optional[str] = None

    def copy(self) -> "BerkyLiveHandler":
        """Create a fresh handler for a new stream session."""
        return BerkyLiveHandler(movement_manager=self._movement_manager, interaction_mode=self._interaction_mode)

    async def start_up(self) -> None:
        """Connect to LLM Engine before audio starts flowing."""
        if self._connected:
            return
        await self.client.connect()
        self._connected = True
        logger.info("Berky live handler connected to LLM Engine")

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
        asyncio.create_task(self._speak_thinking_ack())

        async def _watchdog() -> None:
            await asyncio.sleep(THINKING_TIMEOUT_SECONDS)
            if self._thinking_token == token:
                self._end_thinking()

        asyncio.create_task(_watchdog())

    async def _speak_thinking_ack(self) -> None:
        """Speak a short local acknowledgment right away, alongside the thinking motion.

        Deliberately bypasses llm_engine entirely (no backend round trip) - see
        THINKING_ACK_LINES.
        """
        line = random.choice(THINKING_ACK_LINES)
        try:
            synth = await self.tts.synthesize(line)
        except Exception:
            logger.warning("Failed to synthesize thinking-ack line", exc_info=True)
            return
        if synth is None:
            return
        self._speaking = True
        try:
            await self.output_queue.put(synth)
        finally:
            self._speaking = False

    def _end_thinking(self) -> None:
        """Stop head motion - called right before real audio starts, or by the watchdog."""
        if self._thinking_stop is not None:
            self._thinking_stop()
            self._thinking_stop = None

    async def _synth_worker(self) -> None:
        """Continuously synthesize sentences off _chunk_queue into _synth_queue.

        Runs independently of playback pacing - see the lookahead note in __init__ - so
        synthesis of the next sentence starts as soon as it's dequeued, not once the
        previous sentence finishes playing.
        """
        while True:
            sentence = await self._chunk_queue.get()
            try:
                synth = await self.tts.synthesize(sentence)
            except Exception:
                logger.warning("Failed to synthesize streamed chunk %r", sentence, exc_info=True)
                synth = None
            finally:
                self._chunk_queue.task_done()
            await self._synth_queue.put(synth)

    async def _chunk_worker(self) -> None:
        """Background consumer: play synthesized sentences off _synth_queue one at a time, in order.

        The head nods subtly in sync with each chunk's own audio (see moves.nod_along_with_audio);
        the thinking motion happens beforehand, while the answer is still being generated.
        """
        while True:
            synth = await self._synth_queue.get()
            try:
                if synth is not None:
                    sample_rate, samples = synth
                    await self.output_queue.put(synth)
                    await asyncio.to_thread(nod_along_with_audio, self._movement_manager, samples, sample_rate)
            finally:
                self._synth_queue.task_done()

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
            self._end_thinking()  # real audio is about to start
            self._active_request_id = request_id
            self._streaming_active = True
            # Mute the mic for the duration of playback - see _on_agent_message for why.
            self.transcriber.flush()
            self._speaking = True
            if self._synth_worker_task is None or self._synth_worker_task.done():
                self._synth_worker_task = asyncio.create_task(self._synth_worker(), name="chunk-synth")
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
            await self._synth_queue.join()
            self._speaking = False
            self._streaming_active = False

    async def _on_agent_message(self, message: dict[str, Any]) -> None:
        """Speak and display one agent message.

        Synthesizes to raw samples and enqueues them as a plain tuple, which
        console.py's play_loop recognizes and pushes through robot.media -
        the robot's own speaker - rather than calling self.tts.speak(), which
        would play through this host machine's local audio output instead.

        If this response already streamed sentence-by-sentence via
        _on_answer_chunk, it's already been fully spoken - just update the chat
        display. Otherwise (streaming didn't fire, e.g. an older agent or an
        error), fall back to synthesizing the whole text here, still
        sentence-by-sentence with one-sentence lookahead so playback starts on
        the first sentence rather than waiting for the entire response.
        """
        text = _message_text(message)
        if not text:
            return

        if self._streamed_last_response:
            self._streamed_last_response = False
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
            return

        chunks = split_into_speech_sentences(text)
        if not chunks:
            return

        self._end_thinking()  # real audio is about to start

        # Mute the mic for the duration of playback - otherwise Berky's own
        # voice, played through the robot's speaker, bleeds back into its mic
        # and gets transcribed as if it were something the user said (observed
        # live: repeated "Thank you." transcripts arriving while a response
        # was still being spoken). Flush any in-progress buffer first so a
        # partial segment doesn't carry across the mute boundary.
        self.transcriber.flush()
        self._speaking = True
        any_audio = False
        try:
            next_synth = asyncio.ensure_future(self.tts.synthesize(chunks[0]))
            for i in range(len(chunks)):
                synth = await next_synth
                if i + 1 < len(chunks):
                    next_synth = asyncio.ensure_future(self.tts.synthesize(chunks[i + 1]))
                if synth is None:
                    continue
                any_audio = True
                sample_rate, samples = synth
                await self.output_queue.put(synth)
                await asyncio.to_thread(nod_along_with_audio, self._movement_manager, samples, sample_rate)
        finally:
            self._speaking = False

        if not any_audio:
            # No file-capable TTS binary found; fall back to direct playback
            # on this machine so the response is at least audible somewhere.
            await self.tts.speak(text)

        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))

    async def receive(self, frame: Tuple[int, NDArray[Any]]) -> None:
        """Accept browser microphone frames and send completed transcripts."""
        if not self._connected:
            return
        if self._speaking:
            # Mic muted while Berky is talking - see _on_agent_message.
            return
        if self._interaction_mode is not None and not self._interaction_mode.is_community_assistant():
            # Welcomer profile is active - see interaction_mode.py. Don't bother
            # transcribing mic audio at all while it's Berkie's turn to run.
            return

        sample_rate, audio = frame
        transcript = await self.transcriber.accept(sample_rate, audio)

        if not transcript:
            return

        logger.info("Browser transcript: %s", transcript)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        if contains_wake_phrase(transcript, config.BERKY_WAKE_PHRASE or ""):
            self._begin_thinking()
        try:
            # last_speaker comes from diarization (see LocalWhisperSegmenter),
            # if enabled - previously never passed through here at all, so
            # the diarized label was computed and then silently discarded;
            # llm_engine never saw it and couldn't use it for speaker-count
            # questions.
            await self.client.send_transcript(
                transcript,
                final=True,
                speaker=self.transcriber.last_speaker,
            )
        except Exception:
            logger.warning("Failed to send transcript — LLM Engine disconnected, will retry on reconnect")

    async def emit(self) -> AdditionalOutputs | Tuple[int, NDArray[Any]] | None:
        """Emit chatbot updates when they are available."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Disconnect from LLM Engine and clear pending output."""
        if self._chunk_worker_task is not None and not self._chunk_worker_task.done():
            self._chunk_worker_task.cancel()
        if self._synth_worker_task is not None and not self._synth_worker_task.done():
            self._synth_worker_task.cancel()
        try:
            await self.client.disconnect()
        finally:
            self._connected = False
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
