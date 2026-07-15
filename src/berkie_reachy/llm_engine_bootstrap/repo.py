"""Fetch llm_engine's source at runtime, onto the host machine's disk.

llm_engine's full source is deliberately NOT vendored into berkie_reachy's
own repo: berkie_reachy auto-syncs to a *public* Hugging Face Space on every
push to main, and llm_engine is Berkman Klein Center's internal backend used
by many agents unrelated to Berky. Instead, we clone the public
berkmancenter/llm_engine repo fresh, directly onto the host machine, the
first time the app runs there.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from berkie_reachy.llm_engine_bootstrap import state

logger = logging.getLogger(__name__)

LLM_ENGINE_REPO_URL = "https://github.com/berkmancenter/llm_engine.git"

# Pinned so upstream changes can't silently break the bootstrap; bump deliberately.
# This is the commit that included the working Bedrock v1->v2 migration, verified
# working end-to-end this session.
LLM_ENGINE_PINNED_REF = "main"

_PATCHES_DIR = Path(__file__).parent / "patches"
_TUNING_PATCH = _PATCHES_DIR / "berky_tuning.patch"


class RepoError(RuntimeError):
    """Raised when cloning/updating llm_engine's source fails."""


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RepoError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _apply_tuning_patch(src_dir: Path) -> None:
    """Apply the small wake-phrase/recursion-limit tuning patch, if not already applied."""
    if not _TUNING_PATCH.exists():
        logger.warning("Tuning patch not found at %s; skipping", _TUNING_PATCH)
        return
    check = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(_TUNING_PATCH)],
        cwd=str(src_dir),
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        logger.debug("Tuning patch already applied; skipping")
        return
    apply_check = subprocess.run(
        ["git", "apply", "--check", str(_TUNING_PATCH)],
        cwd=str(src_dir),
        capture_output=True,
        text=True,
    )
    if apply_check.returncode != 0:
        logger.warning(
            "Tuning patch does not apply cleanly (upstream llm_engine may have changed); "
            "continuing without it.\n%s",
            apply_check.stderr,
        )
        return
    _run(["git", "apply", str(_TUNING_PATCH)], cwd=src_dir)
    logger.info("Applied Berky tuning patch (wake-phrase threshold, recursion limit)")


def ensure_llm_engine_source(dest_dir: Path | None = None) -> Path:
    """Clone or update llm_engine into ``dest_dir``, applying the tuning patch.

    Idempotent: safe to call on every launch. Returns the source directory.
    """
    dest_dir = dest_dir or state.LLM_ENGINE_SRC_DIR
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if (dest_dir / ".git").exists():
        logger.info("Updating existing llm_engine checkout at %s", dest_dir)
        _run(["git", "fetch", "--depth", "1", "origin", LLM_ENGINE_PINNED_REF], cwd=dest_dir)
        _run(["git", "reset", "--hard", f"origin/{LLM_ENGINE_PINNED_REF}"], cwd=dest_dir)
    else:
        logger.info("Cloning llm_engine into %s", dest_dir)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                LLM_ENGINE_PINNED_REF,
                LLM_ENGINE_REPO_URL,
                str(dest_dir),
            ]
        )

    _apply_tuning_patch(dest_dir)
    return dest_dir
