"""Local Whisper speech segmentation for Berky."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from scipy.signal import resample
from numpy.typing import NDArray

from berkie_reachy.config import config


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def _mono_float32(frame: NDArray[Any]) -> NDArray[np.float32]:
    audio = np.asarray(frame)
    raw_dtype = audio.dtype
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    audio = audio.astype(np.float32, copy=False)
    if np.issubdtype(raw_dtype, np.integer):
        max_value = float(np.iinfo(raw_dtype).max)
        if max_value > 0:
            audio = audio / max_value
    return audio.reshape(-1)


def _resample_if_needed(audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    output_len = max(1, int(len(audio) * TARGET_SAMPLE_RATE / sample_rate))
    return resample(audio, output_len).astype(np.float32, copy=False)


class LocalWhisperSegmenter:
    """Convert incoming audio frames into finalized transcript chunks."""

    def __init__(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the berky_voice extra or faster-whisper to use local transcription: "
                "`pip install -e .[berky_voice]`."
            ) from exc

        self.model = WhisperModel(
            config.BERKY_WHISPER_MODEL,
            device=config.BERKY_WHISPER_DEVICE,
            compute_type=config.BERKY_WHISPER_COMPUTE_TYPE,
        )
        self.threshold = float(config.BERKY_SPEECH_RMS_THRESHOLD)
        self.silence_samples_required = int(float(config.BERKY_SILENCE_SECONDS) * TARGET_SAMPLE_RATE)
        self.max_samples = int(float(config.BERKY_TRANSCRIBE_WINDOW_SECONDS) * TARGET_SAMPLE_RATE)
        self.min_samples = int(0.45 * TARGET_SAMPLE_RATE)

        self._buffer: list[NDArray[np.float32]] = []
        self._active = False
        self._speech_samples = 0
        self._silence_samples = 0

        self.last_speaker: str | None = None
        self._diarizer = None
        if config.BERKY_DIARIZATION_ENABLED:
            try:
                from berkie_reachy.diarization import Diarizer
                self._diarizer = Diarizer(
                    hf_token=config.HF_TOKEN,
                    device=config.BERKY_DIARIZATION_DEVICE,
                )
                logger.info("Speaker diarization enabled (device=%s)", config.BERKY_DIARIZATION_DEVICE)
            except ImportError as exc:
                logger.warning("Diarization requested but pyannote.audio not installed: %s", exc)

    async def accept(self, sample_rate: int, frame: NDArray[Any]) -> str | None:
        """Accept one audio frame and return a transcript when a segment ends."""
        audio = _resample_if_needed(_mono_float32(frame), sample_rate)
        if len(audio) == 0:
            return None

        rms = float(np.sqrt(np.mean(np.square(audio))))
        is_speech = rms >= self.threshold

        if is_speech:
            self._active = True
            self._silence_samples = 0

        if not self._active:
            return None

        self._buffer.append(audio)
        self._speech_samples += len(audio)
        if is_speech:
            self._silence_samples = 0
        else:
            self._silence_samples += len(audio)

        long_enough = self._speech_samples >= self.min_samples
        silence_done = self._silence_samples >= self.silence_samples_required
        maxed = self._speech_samples >= self.max_samples
        if not ((long_enough and silence_done) or maxed):
            return None

        segment = self._consume_buffer()
        if segment is None:
            return None
        return await asyncio.to_thread(self._transcribe, segment)

    def flush(self) -> NDArray[np.float32] | None:
        """Return pending audio, if any, and reset state."""
        return self._consume_buffer()

    @property
    def is_active(self) -> bool:
        """Whether the segmenter is currently inside a speech segment."""
        return self._active

    def _consume_buffer(self) -> NDArray[np.float32] | None:
        if not self._buffer:
            self._reset()
            return None
        segment = np.concatenate(self._buffer).astype(np.float32, copy=False)
        self._reset()
        if len(segment) < self.min_samples:
            return None
        return segment

    def _reset(self) -> None:
        self._buffer = []
        self._active = False
        self._speech_samples = 0
        self._silence_samples = 0

    def _transcribe(self, audio: NDArray[np.float32]) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        segments = list(segments)  # materialise so diarizer can reuse the audio

        if self._diarizer is not None:
            aligned = self._diarizer.align_with_asr(segments, audio, TARGET_SAMPLE_RATE)
            if aligned:
                totals: dict[str, int] = {}
                for sp, t in aligned:
                    totals[sp] = totals.get(sp, 0) + len(t)
                self.last_speaker = max(totals, key=totals.__getitem__)
                text = " ".join(t for _, t in aligned)
            else:
                self.last_speaker = None
                text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        else:
            self.last_speaker = None
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        transcript = " ".join(text.split())
        if transcript:
            logger.info("Whisper transcript [%s]: %s", self.last_speaker or "unknown", transcript)
        return transcript
