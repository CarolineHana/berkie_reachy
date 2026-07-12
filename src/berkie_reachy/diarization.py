"""Speaker diarization for Berkie Reachy."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment

logger = logging.getLogger(__name__)

_MODEL_ID = "pyannote/speaker-diarization-3.1"


class Diarizer:
    """Lazy-loads the pyannote pipeline on first use to avoid startup cost."""

    def __init__(self, hf_token: Optional[str] = None, device: str = "cpu") -> None:
        self._token = hf_token
        self._device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Run: pip install 'pyannote.audio>=3.3.0' soundfile"
            ) from exc

        logger.info("Loading %s (first call)…", _MODEL_ID)
        self._pipeline = Pipeline.from_pretrained(_MODEL_ID, token=self._token)
        if self._device == "cuda":
            import torch  # type: ignore
            self._pipeline = self._pipeline.to(torch.device("cuda"))
        logger.info("pyannote pipeline ready.")

    def _run(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float, str]]:
        self._load()
        import soundfile as sf  # type: ignore
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        try:
            result = self._pipeline({"uri": "seg", "audio": buf})
        except Exception:
            logger.exception("pyannote diarization failed")
            return []
        # pyannote 4.x wraps output in DiarizeOutput; 3.x returns Annotation directly
        annotation = getattr(result, "speaker_diarization", result)
        return [(t.start, t.end, sp) for t, _, sp in annotation.itertracks(yield_label=True)]

    def dominant_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Return the label of the speaker who talked longest in this chunk."""
        segs = self._run(audio, sample_rate)
        if not segs:
            return "SPEAKER_00"
        totals: dict[str, float] = {}
        for start, end, sp in segs:
            totals[sp] = totals.get(sp, 0.0) + (end - start)
        return max(totals, key=totals.__getitem__)

    def align_with_asr(
        self,
        asr_segments: list["Segment"],
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[tuple[str, str]]:
        """Align pyannote speaker turns with faster-whisper Segment objects.

        Returns [(speaker_label, text), …] — one entry per diarization turn
        that overlaps with ASR text. Falls back to a single anonymous turn
        if diarization returns nothing.
        """
        diarization = self._run(audio, sample_rate)
        if not diarization:
            full = " ".join(s.text.strip() for s in asr_segments if s.text.strip())
            return [("SPEAKER_00", full)] if full else []

        result: list[tuple[str, str]] = []
        for start, end, speaker in diarization:
            words = [
                seg.text.strip()
                for seg in asr_segments
                if seg.start < end and seg.end > start and seg.text.strip()
            ]
            text = " ".join(words).strip()
            if text:
                result.append((speaker, text))
        return result
