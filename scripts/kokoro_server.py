#!/usr/bin/env python3
"""Persistent Kokoro TTS server — loads pipeline once, serves speak requests over HTTP."""

import json
import sys
import numpy as np
import sounddevice as sd
from http.server import BaseHTTPRequestHandler, HTTPServer
from kokoro import KPipeline

PORT = 15731
SAMPLE_RATE = 24000
DEVICE_NAME = "Reachy Mini Audio"


def _find_device() -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if DEVICE_NAME in d["name"] and d["max_output_channels"] > 0:
            return i
    return None


print("Loading Kokoro pipeline...", flush=True)
_pipeline = KPipeline(lang_code="b")
print("Kokoro ready.", flush=True)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        text = json.loads(body).get("text", "").strip()
        if text:
            samples = [audio for _, _, audio in _pipeline(text, voice="bm_fable")]
            if samples:
                audio = np.concatenate(samples).astype(np.float32)
                audio = np.clip(audio * 3.0, -1.0, 1.0)
                sd.play(audio, samplerate=SAMPLE_RATE, device=_find_device(), blocking=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), _Handler)
    print(f"Kokoro TTS server listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.", flush=True)
        sys.exit(0)
