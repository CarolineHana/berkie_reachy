"""Shared toggle between the Welcomer and Community Assistant interaction profiles.

Per the launch-event spec: "Berkie switches profiles by location/phase; it does not run
both logics at once." Both features' underlying loops/handlers (Welcomer's camera-polling
thread, BerkyLiveHandler's mic pipeline) stay alive the whole time - this object just gates
whether each one is allowed to actually act (speak / respond to a question), so switching
is instant and doesn't require tearing down camera or socket connections.
"""

from __future__ import annotations
import threading


class InteractionMode:
    """Thread-safe current-mode holder, read by both features' worker loops."""

    WELCOMER = "welcomer"
    COMMUNITY_ASSISTANT = "community_assistant"

    def __init__(self, initial: str = COMMUNITY_ASSISTANT) -> None:
        """Start in ``initial`` mode (default: Community Assistant)."""
        self._lock = threading.Lock()
        self._mode = initial

    @property
    def mode(self) -> str:
        """Current mode string - one of WELCOMER or COMMUNITY_ASSISTANT."""
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        with self._lock:
            self._mode = value

    def is_welcomer(self) -> bool:
        """Return True when Welcomer should be active."""
        return self.mode == self.WELCOMER

    def is_community_assistant(self) -> bool:
        """Return True when the Community Assistant (mic Q&A) should be active."""
        return self.mode == self.COMMUNITY_ASSISTANT
