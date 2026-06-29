"""Text-to-speech helpers for Berky."""

from __future__ import annotations

import shlex
import shutil
import asyncio
import logging
import platform
import subprocess

from berkie_reachy.config import config


logger = logging.getLogger(__name__)


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


class CommandTTS:
    """Speak text by running a local TTS command."""

    def __init__(self, command_template: str | None = None) -> None:
        self.command_template = command_template or _default_tts_command()

    async def speak(self, text: str) -> None:
        """Speak one utterance, if a local command is available."""
        clean_text = " ".join(text.split())
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

    def _argv(self, text: str) -> list[str]:
        parts = shlex.split(self.command_template or "")
        if not parts:
            raise RuntimeError("TTS command template is empty.")
        replaced = [part.replace("{text}", text) for part in parts]
        if all("{text}" not in part for part in parts):
            replaced.append(text)
        return replaced
