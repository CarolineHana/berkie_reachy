"""Standalone speaker-diarization HTTP server.

Runs inside its own isolated venv (created by ``diarization_bootstrap``) that
has pyannote.audio installed - never imported from berkie_reachy's own
process, since pyannote's dependency tree bumps protobuf to a version that
breaks mediapipe (used for head tracking). This file has no dependency on
the berkie_reachy package at all so it can be copied/run standalone from
inside that isolated venv; talk to it over the tiny local HTTP/JSON protocol
below instead.
"""

from __future__ import annotations

import io
import json
import base64
import logging
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger("diarization_server")

_MODEL_ID = "pyannote/speaker-diarization-3.1"


class PipelineHolder:
    """Lazy-loads the pyannote pipeline on first real request."""

    def __init__(self, hf_token: str | None, device: str) -> None:
        self._token = hf_token
        self._device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline

        logger.info("Loading %s (first request)...", _MODEL_ID)
        self._pipeline = Pipeline.from_pretrained(_MODEL_ID, token=self._token)
        if self._device == "cuda":
            import torch

            self._pipeline = self._pipeline.to(torch.device("cuda"))
        logger.info("pyannote pipeline ready.")

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float, str]]:
        self._load()
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        result = self._pipeline({"uri": "seg", "audio": buf})
        # pyannote 4.x wraps output in DiarizeOutput; 3.x returns Annotation directly
        annotation = getattr(result, "speaker_diarization", result)
        return [(t.start, t.end, sp) for t, _, sp in annotation.itertracks(yield_label=True)]


def _align_with_asr(
    pipeline_holder: PipelineHolder,
    segments: list[dict],
    audio: np.ndarray,
    sample_rate: int,
) -> list[tuple[str, str]]:
    """Align pyannote speaker turns with ASR segment dicts ({start, end, text})."""
    diarization = pipeline_holder.diarize(audio, sample_rate)
    if not diarization:
        full = " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())
        return [("SPEAKER_00", full)] if full else []

    result: list[tuple[str, str]] = []
    for start, end, speaker in diarization:
        words = [
            s["text"].strip()
            for s in segments
            if s["start"] < end and s["end"] > start and s.get("text", "").strip()
        ]
        text = " ".join(words).strip()
        if text:
            result.append((speaker, text))
    return result


def _make_handler(pipeline_holder: PipelineHolder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            logger.debug(format, *args)

        def _respond(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/align":
                self._respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                pcm = np.frombuffer(base64.b64decode(body["audio_b64"]), dtype=np.int16)
                audio = pcm.astype(np.float32) / 32768.0
                sample_rate = int(body["sample_rate"])
                aligned = _align_with_asr(pipeline_holder, body["segments"], audio, sample_rate)
                self._respond(200, {"aligned": aligned})
            except Exception:
                logger.exception("Diarization request failed")
                self._respond(500, {"error": "diarization_failed"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated diarization HTTP server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    pipeline_holder = PipelineHolder(args.hf_token, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(pipeline_holder))
    logger.info("Diarization server listening on 127.0.0.1:%s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
