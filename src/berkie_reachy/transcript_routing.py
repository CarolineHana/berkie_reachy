"""Route a transcript to the right conversation channel based on its content.

Berky's conversation has two question-answering agents listening on
different channels: voiceAssistant on "transcript" (general Q&A) and
eventHistorian on "historian" (searches this event's own past transcripts,
scoped to this conversation's topic via agentConfig.topicIds). This module
picks which one a given question should go to.
"""

from __future__ import annotations
import re


HISTORIAN_CHANNEL = "historian"

# Conservative on purpose: default to voiceAssistant (the primary assistant)
# unless the question clearly references something from earlier/past
# sessions rather than asking something fresh.
_HISTORY_PATTERN = re.compile(
    r"\b("
    r"earlier|before|previous|last time|past event|past session|"
    r"history|remember|recap|summar\w*|"
    r"first thing|did i ask|did you (already|previously)|"
    r"what happened|what did (i|we)|archive"
    r")\b",
    re.IGNORECASE,
)


def classify_channel(text: str) -> str | None:
    """Return the channel name a transcript should be routed to.

    Returns ``None`` for the default transcript channel (voiceAssistant),
    or ``HISTORIAN_CHANNEL`` when the question looks like it's asking about
    something from earlier/past sessions.
    """
    if _HISTORY_PATTERN.search(text):
        return HISTORIAN_CHANNEL
    return None


__all__ = ["HISTORIAN_CHANNEL", "classify_channel"]
