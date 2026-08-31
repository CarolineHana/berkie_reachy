"""Text-to-speech helpers for Berky."""

from __future__ import annotations

import re
import wave
import shlex
import shutil
import asyncio
import logging
import platform
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from berkie_reachy.config import config


logger = logging.getLogger(__name__)


def _clean_for_speech(text: str) -> str:
    """Strip markdown and symbols that sound bad when read aloud."""
    # Remove markdown links [label](url) → label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Replace em dash and en dash with a pause comma
    text = re.sub(r'[—–]', ',', text)
    # Remove other markdown symbols: #, >, |, ~, `
    text = re.sub(r'[#>`|~]', '', text)
    # Remove backtick code spans
    text = re.sub(r'`[^`]*`', '', text)
    # Collapse multiple spaces/punctuation
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
# Below this, merge a fragment into the next one rather than emitting it as its
# own TTS call - each call pays a fixed subprocess-startup cost, so splitting
# too finely (e.g. on every short "Yes." / "No.") trades that overhead for no
# real latency benefit.
_MIN_CHUNK_CHARS = 40


def split_into_speech_sentences(text: str) -> list[str]:
    """Split cleaned text into sentence-sized chunks for incremental TTS playback.

    Lets the caller start synthesizing/playing the first chunk without
    waiting for the entire (possibly multi-sentence) response to render.
    """
    text = text.strip()
    if not text:
        return []
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not raw_parts:
        return [text]

    chunks: list[str] = []
    buffer = ""
    for part in raw_parts:
        buffer = f"{buffer} {part}".strip() if buffer else part
        if len(buffer) >= _MIN_CHUNK_CHARS:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {buffer}".strip()
        else:
            chunks.append(buffer)
    return chunks


def _default_tts_command() -> str | None:
    if config.BERKY_TTS_COMMAND:
        return config.BERKY_TTS_COMMAND
    if platform.system() == "Darwin" and shutil.which("say"):
        return "say {text}"
    if shutil.which("espeak-ng"):
        return "espeak-ng {text}"
    if shutil.which("espeak"):
        return "espeak {text}"
    return None


def _synth_to_file_argv(out_path: str) -> list[str] | None:
    """Build an argv that renders speech to ``out_path`` as a mono 16-bit WAV, if possible.

    Deliberately separate from the direct-play command template: routing audio to
    the robot's speaker (via ``robot.media.push_audio_sample``) needs raw samples,
    which means synthesizing to a file we can read back rather than letting the
    TTS binary play straight to this machine's own audio output.
    """
    if platform.system() == "Darwin" and shutil.which("say"):
        return ["say", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", out_path]
    if shutil.which("espeak-ng"):
        return ["espeak-ng", "-w", out_path]
    if shutil.which("espeak"):
        return ["espeak", "-w", out_path]
    return None


class CommandTTS:
    """Speak text by running a local TTS command."""

    def __init__(self, command_template: str | None = None) -> None:
        self.command_template = command_template or _default_tts_command()

    async def synthesize(self, text: str) -> Optional[Tuple[int, NDArray[Any]]]:
        """Render ``text`` to audio samples instead of playing them on this machine.

        Returns ``(sample_rate, int16_samples)`` for the caller to push through the
        robot's own speaker, or ``None`` if no file-capable TTS binary is available.
        """
        clean_text = " ".join(_clean_for_speech(text).split())
        if not clean_text:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = str(Path(tmp_dir) / "berky_tts.wav")
            argv = _synth_to_file_argv(out_path)
            if argv is None:
                return None
            argv = argv + [clean_text]
            logger.info("Synthesizing agent response with %s", argv[0])
            result = await asyncio.to_thread(
                subprocess.run,
                argv,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0 or not Path(out_path).exists():
                logger.warning("TTS synthesis failed (exit %s) for %s", result.returncode, argv[0])
                return None

            with wave.open(out_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                raw = wav_file.readframes(n_frames)
                sampwidth = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()

            if sampwidth != 2:
                logger.warning("Unexpected TTS sample width %s; expected 16-bit PCM", sampwidth)
                return None

            samples = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels)[:, 0]
            return sample_rate, samples

    async def speak(self, text: str) -> None:
        """Speak one utterance, if a local command is available."""
        clean_text = " ".join(_clean_for_speech(text).split())
        if not clean_text:
            return
        if not self.command_template:
            logger.warning("No TTS command configured; agent said: %s", clean_text)
            return

        argv = self._argv(clean_text)
        logger.info("Speaking agent response with %s", argv[0])
        await asyncio.to_thread(
            subprocess.run,
            argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def speak_chunked(self, text: str) -> None:
        """Speak text sentence-by-sentence instead of as one clip.

        Playback of the first sentence can start as soon as it's synthesized,
        rather than waiting for the entire (possibly multi-sentence) response
        to render - the dominant lever for cutting time-to-first-audio on
        longer responses.
        """
        if not self.command_template:
            clean_text = " ".join(_clean_for_speech(text).split())
            if clean_text:
                logger.warning("No TTS command configured; agent said: %s", clean_text)
            return
        for chunk in split_into_speech_sentences(text):
            await self.speak(chunk)

    def _argv(self, text: str) -> list[str]:
        parts = shlex.split(self.command_template or "")
        if not parts:
            raise RuntimeError("TTS command template is empty.")
        replaced = [part.replace("{text}", text) for part in parts]
        if all("{text}" not in part for part in parts):
            replaced.append(text)
        return replaced


def speak_sync_through_robot(text: str, tts: CommandTTS, robot: Any, movement_manager: Any = None) -> None:
    """Synthesize ``text`` and play it through the robot's speaker, blocking the caller.

    For plain-thread callers (e.g. welcomer.py) that aren't running inside the app's own
    asyncio event loop - runs CommandTTS.synthesize() in a short-lived loop of its own.
    Mirrors berky_reachy.py's _play_through_robot, minus the streaming/chunking machinery
    that's unnecessary for a single short greeting line.

    If ``movement_manager`` is given, nods along with the audio (see moves.nod_along_with_audio)
    instead of just holding still for playback - the same subtle talking-nod used elsewhere.
    """
    import asyncio as _asyncio

    from fastrtc import audio_to_float32
    from scipy.signal import resample

    async def _do_speak() -> None:
        synth = await tts.synthesize(text)
        if synth is None:
            await tts.speak(text)
            return

        sample_rate, samples = synth
        audio_frame = audio_to_float32(samples)
        output_sample_rate = robot.media.get_output_audio_samplerate()
        if sample_rate != output_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * output_sample_rate / sample_rate))
            sample_rate = output_sample_rate

        robot.media.push_audio_sample(audio_frame)
        if movement_manager is not None:
            from berkie_reachy.moves import nod_along_with_audio

            nod_along_with_audio(movement_manager, audio_frame, sample_rate)
        else:
            await _asyncio.sleep(len(audio_frame) / sample_rate)

    _asyncio.run(_do_speak())
