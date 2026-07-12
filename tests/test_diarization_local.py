"""
Quick local test for the diarization pipeline.

Records N seconds from your laptop mic, runs it through LocalWhisperSegmenter
(with diarization enabled), and prints the transcript + speaker label.
No Reachy robot or llm_engine connection needed.

Usage:
    # Record from mic (default 10 seconds):
    BERKY_DIARIZATION_ENABLED=true python tests/test_diarization_local.py

    # Pass a WAV/MP3 file instead:
    BERKY_DIARIZATION_ENABLED=true python tests/test_diarization_local.py path/to/audio.wav

    # Change recording duration:
    BERKY_DIARIZATION_ENABLED=true python tests/test_diarization_local.py --seconds 15
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the src package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

SAMPLE_RATE = 16000


def record_from_mic(seconds: int) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice not installed — run: pip install sounddevice")

    print(f"Recording {seconds}s from microphone... speak now!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording done.\n")
    return audio.reshape(-1)


def load_audio_file(path: str) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("soundfile not installed — run: pip install soundfile")

    try:
        import resampy
        _has_resampy = True
    except ImportError:
        _has_resampy = False

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        if not _has_resampy:
            sys.exit(f"File is {sr}Hz but 16000Hz required. Install resampy: pip install resampy")
        import resampy
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    return audio


def run(audio: np.ndarray) -> None:
    from berkie_reachy.local_whisper import LocalWhisperSegmenter

    segmenter = LocalWhisperSegmenter()

    if segmenter._diarizer is None:
        print("WARNING: diarizer not loaded — is BERKY_DIARIZATION_ENABLED=true set?")

    print("Transcribing...")
    transcript = segmenter._transcribe(audio)

    print("-" * 50)
    print(f"Speaker : {segmenter.last_speaker or '(diarization off)'}")
    print(f"Transcript: {transcript or '(empty)'}")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local diarization test")
    parser.add_argument("file", nargs="?", help="Audio file to process (WAV/MP3/etc). Omit to record from mic.")
    parser.add_argument("--seconds", type=int, default=10, help="Recording duration in seconds (mic mode only)")
    args = parser.parse_args()

    if args.file:
        print(f"Loading {args.file}...")
        audio = load_audio_file(args.file)
    else:
        audio = record_from_mic(args.seconds)

    run(audio)


if __name__ == "__main__":
    main()
