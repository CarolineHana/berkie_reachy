"""Speaker diarization client for Berkie Reachy.

Talks to an isolated diarization server (see diarization_bootstrap/) over a
tiny local HTTP/JSON protocol rather than importing pyannote.audio directly -
that dependency tree bumps protobuf to a version incompatible with mediapipe
(used for optional head tracking), so pyannote is fully isolated into its own
venv+subprocess instead of sharing berkie_reachy's own environment.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment

logger = logging.getLogger(__name__)


class Diarizer:
    """Aligns ASR segments with speaker turns via the isolated diarization server."""

    def __init__(self, hf_token: Optional[str] = None, device: str = "cpu") -> None:
        from berkie_reachy.diarization_bootstrap import ensure_diarization_stack

        # Provisions (or reuses) the isolated venv+server on first construction;
        # returns None (never raises) if that fails for any reason.
        self._base_url = ensure_diarization_stack(hf_token=hf_token, device=device)

    def _run(self, audio: np.ndarray, sample_rate: int, segments: list[dict]) -> list[tuple[str, str]]:
        if not self._base_url:
            return []
        try:
            import httpx

            audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            payload = {
                "audio_b64": base64.b64encode(audio_i16.tobytes()).decode("ascii"),
                "sample_rate": sample_rate,
                "segments": segments,
            }
            resp = httpx.post(f"{self._base_url}/align", json=payload, timeout=30.0)
            resp.raise_for_status()
            return [(sp, text) for sp, text in resp.json()["aligned"]]
        except Exception:
            logger.exception("Diarization request failed")
            return []

    def align_with_asr(
        self,
        asr_segments: list["Segment"],
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[tuple[str, str]]:
        """Align speaker turns with faster-whisper Segment objects.

        Returns [(speaker_label, text), …] — one entry per diarization turn
        that overlaps with ASR text. Falls back to an empty list (caller
        already treats that the same as "diarization unavailable") if the
        isolated server can't be reached.
        """
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in asr_segments]
        return self._run(audio, sample_rate, segments)
