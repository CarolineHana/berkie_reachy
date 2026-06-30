#!/usr/bin/env python3
"""Kokoro TTS client — sends text to the persistent server, falls back to direct if server is down."""

import json
import sys
import urllib.request
import urllib.error

PORT = 15731


def _speak_via_server(text: str) -> bool:
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=120)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _speak_direct(text: str) -> None:
    import numpy as np
    import sounddevice as sd
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="b")
    samples = [audio for _, _, audio in pipeline(text, voice="bm_fable")]
    if not samples:
        return
    audio = np.concatenate(samples).astype(np.float32)
    audio = np.clip(audio * 3.0, -1.0, 1.0)
    device = None
    for i, d in enumerate(sd.query_devices()):
        if "Reachy Mini Audio" in d["name"] and d["max_output_channels"] > 0:
            device = i
            break
    sd.play(audio, samplerate=24000, device=device, blocking=True)


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        return
    if not _speak_via_server(text):
        _speak_direct(text)


if __name__ == "__main__":
    main()
