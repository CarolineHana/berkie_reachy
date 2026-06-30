#!/usr/bin/env python3
"""Kokoro TTS runner for Berkie — reads text from argv and plays through Reachy Mini Audio."""

import sys
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEVICE_NAME = "Reachy Mini Audio"


def _find_device() -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if DEVICE_NAME in d["name"] and d["max_output_channels"] > 0:
            return i
    return None


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        return

    pipeline = KPipeline(lang_code="b")
    samples = [audio for _, _, audio in pipeline(text, voice="bm_fable")]
    if not samples:
        return

    audio = np.concatenate(samples).astype(np.float32)
    audio = np.clip(audio * 3.0, -1.0, 1.0)
    device = _find_device()
    sd.play(audio, samplerate=SAMPLE_RATE, device=device, blocking=True)


if __name__ == "__main__":
    main()
