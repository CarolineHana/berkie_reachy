"""Feature A - Welcomer: vision-triggered, shallow, high-throughput greetings.

Spec: see the "Berkie Launch Event Requirements" doc. Runs the registration-table
persona - the same character as everywhere else in the system, dialed down (low
playfulness, low verbosity - "clipped" rather than "turned up").

Reuses the CameraWorker's already-running frame buffer rather than opening its own
camera stream, and stays independent of any particular TTS/robot wiring - callers
supply a `speak()` callback, so this module is testable without hardware.
"""

from __future__ import annotations
import time
import random
import logging
import threading
from typing import Any, Callable, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# Five-axis personality config for this feature (see the Personality Lab spec).
# Not consumed programmatically today - the line pool and pacing below are the
# concrete expression of these values (low playfulness -> no jokes, low verbosity
# -> <=6 words of new content per line). Kept here so the numbers travel with the
# behavior they justify, and so a future axis-driven generator has something to
# calibrate against.
AXES = {
    "context": "welcomer",
    "warmth": 0.75,
    "playfulness": 0.5,
    "formality": 0.2,
    "verbosity": 0.15,
    "proactivity": 0.5,
}

# Core rotation: generic, always safe, no jokes, <=6 words of new content.
CORE_LINES = [
    "Hi! Welcome to the Berkman Klein Center.",
    "Hey there — welcome to BKC.",
    "Welcome in — glad you found us.",
    "Hi! I'm Berkie — welcome to the Center.",
    "Welcome to BKC — good to have you.",
    "Hello! Great to have you at the Center.",
    "Hi there — make yourself at home at BKC.",
    "Welcome — come on in.",
    "Hi! So glad you're here at BKC.",
    "Welcome to the Center — come on in.",
    "Hey there — welcome, glad you made it.",
    "Hi! Welcome — good to see you at BKC.",
    "Welcome in — you're in the right place.",
    "Hello! Welcome to the Berkman Klein Center.",
    "Hi there — great to have you with us.",
    "Welcome — settle in and enjoy.",
    "Hey! Welcome to the Center.",
    "Hi! Lovely to have you at BKC.",
    "Hey there, welcome to the jungle.",
]

# Launch/orientation flavor: context-wide, always true at this event.
LAUNCH_LINES = [
    "Welcome to launch week at BKC!",
    "Hi! Big year ahead — glad you're here for it.",
    "Welcome — this is where the year begins.",
    "Hi! New to the Center? You're in the right place.",
    "Welcome aboard — the year starts here.",
    "Glad you made it — launch week's just getting going.",
]

# A single warm line for when several people arrive as a cluster - Berkie doesn't
# stutter through N individual greetings for N faces.
GROUP_LINES = [
    "Hi everyone — welcome to BKC!",
    "Hey all — great to see you here.",
    "Welcome, all of you — come on in.",
]

ALL_INDIVIDUAL_LINES = CORE_LINES + LAUNCH_LINES


@dataclass
class WelcomerConfig:
    """Tunable thresholds - see the spec's "Open decisions to lock before launch" list.

    The defaults here are placeholders pending on-site calibration, not measured values.
    """

    # Bounding-box area as a fraction of frame area, used as a distance proxy (no real
    # depth sensor available). Above this -> "close and oriented" (Welcome tier). Needs
    # tuning against the actual registration-table camera position/FOV before launch.
    close_area_fraction: float = 0.0005
    # Below this detector confidence, don't even Glance - treat as noise.
    low_confidence_threshold: float = 0.3
    # One welcome per person per this many minutes.
    no_repeat_window_minutes: float = 20.0
    # Minimum gap between any two spoken greetings, regardless of who they're for -
    # spec calls for a 10-15s delay so greetings don't fire back-to-back.
    min_greeting_gap_seconds: float = 5.0
    # A track not seen for this long is dropped - a re-appearance after this gets a
    # fresh visual ID (and is treated as a new person per the spec's low-confidence
    # fallback: prefer an occasional repeat greeting over a false "weren't you just
    # here?").
    track_expiry_seconds: float = 5.0
    # Centroid-matching distance (in normalized [-1, 1] coords) below which a new
    # detection is considered the same track as an existing one.
    track_match_distance: float = 0.25
    # How often to poll the camera worker's latest frame for presence detection.
    poll_interval_seconds: float = 0.4


@dataclass
class _PersonTrack:
    track_id: int
    center: tuple[float, float]
    last_seen: float
    area_fraction: float
    last_greeted: Optional[float] = None


class Welcomer:
    """Vision-triggered greeter: Glance for distant/passing people, Welcome for close ones.

    Runs its own polling loop (a plain thread, matching CameraWorker's style) reading
    frames from an already-running CameraWorker - it does not touch the camera directly
    and does not steer the head (CameraWorker's own face-tracking already orients toward
    whoever's most prominent; this module only decides *whether to speak*).
    """

    def __init__(
        self,
        camera_worker: Any,
        head_tracker: Any,
        speak: Callable[[str], None],
        config: Optional[WelcomerConfig] = None,
        interaction_mode: Optional[Any] = None,
    ) -> None:
        """Wire up the presence loop against an already-running camera worker."""
        self.camera_worker = camera_worker
        self.head_tracker = head_tracker
        self.speak = speak
        self.config = config or WelcomerConfig()
        self.interaction_mode = interaction_mode

        self._tracks: dict[int, _PersonTrack] = {}
        self._next_track_id = 0
        self._last_greeting_time: float = 0.0
        self._used_individual_lines: list[str] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the presence-polling loop in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True)
        self._thread.start()
        logger.info("Welcomer started")

    def stop(self) -> None:
        """Stop the presence-polling loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Welcomer stopped")

    def _working_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.interaction_mode is None or self.interaction_mode.is_welcomer():
                    frame = self.camera_worker.get_latest_frame()
                    if frame is not None:
                        self._process_frame(frame)
            except Exception:
                logger.exception("Welcomer loop error")
            time.sleep(self.config.poll_interval_seconds)

    def _process_frame(self, frame: Any) -> None:
        now = time.monotonic()
        faces = self.head_tracker.get_all_faces(frame)
        faces = [f for f in faces if f["confidence"] >= self.config.low_confidence_threshold]

        self._expire_stale_tracks(now)
        matched_tracks = [self._match_or_create_track(f, now) for f in faces]

        eligible = [
            t
            for t in matched_tracks
            if t.area_fraction >= self.config.close_area_fraction and self._is_greeting_eligible(t, now)
        ]
        if not eligible:
            return

        if now - self._last_greeting_time < self.config.min_greeting_gap_seconds:
            # Global pacing: hold off even if someone's eligible - don't fire greetings
            # back-to-back. They'll still be eligible next poll (their track persists).
            return

        if len(eligible) > 1:
            line = random.choice(GROUP_LINES)
        else:
            line = self._pick_individual_line()

        logger.info("Welcomer speaking to %d eligible track(s): %r", len(eligible), line)
        try:
            self.speak(line)
        except Exception:
            logger.exception("Welcomer failed to speak greeting")
            return

        self._last_greeting_time = now
        for track in eligible:
            track.last_greeted = now

    def _match_or_create_track(self, face: dict[str, Any], now: float) -> _PersonTrack:
        cx, cy = float(face["center"][0]), float(face["center"][1])
        best_id = None
        best_dist = self.config.track_match_distance
        for track_id, track in self._tracks.items():
            dist = ((track.center[0] - cx) ** 2 + (track.center[1] - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is not None:
            track = self._tracks[best_id]
            track.center = (cx, cy)
            track.last_seen = now
            track.area_fraction = face["area_fraction"]
            return track

        # No confident match - per spec, treat as a new person rather than risk a false
        # re-identification ("weren't you just here?").
        track_id = self._next_track_id
        self._next_track_id += 1
        track = _PersonTrack(track_id=track_id, center=(cx, cy), last_seen=now, area_fraction=face["area_fraction"])
        self._tracks[track_id] = track
        return track

    def _expire_stale_tracks(self, now: float) -> None:
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.config.track_expiry_seconds]
        for tid in stale:
            del self._tracks[tid]

    def _is_greeting_eligible(self, track: _PersonTrack, now: float) -> bool:
        if track.last_greeted is None:
            return True
        return (now - track.last_greeted) >= self.config.no_repeat_window_minutes * 60.0

    def _pick_individual_line(self) -> str:
        # High variety: avoid repeating a line until the whole pool's been used once.
        available = [line for line in ALL_INDIVIDUAL_LINES if line not in self._used_individual_lines]
        if not available:
            self._used_individual_lines = []
            available = ALL_INDIVIDUAL_LINES
        line = random.choice(available)
        self._used_individual_lines.append(line)
        return line
