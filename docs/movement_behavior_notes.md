# Movement Behavior Notes

Notes on Berky's automatic movement behavior and profile tool config, and the
reasoning behind recent changes (2026-07-12/13) — useful context since the
"why" isn't obvious from the code alone.

## Design intent (as clarified by the operator)

Keep the gentle, automatic/idle animations. Remove anything that looks like
"dancing" or a "stark"/fast head movement, whether triggered automatically or
via an explicit tool call. Specifically:

- **Keep**: breathing (idle head bob), face-tracking (but slowed — see
  below), the listening body-yaw sway + ear sway, the speaking head-nod, and
  the `sweep_look` tool.
- **Remove**: the `dance` and `play_emotion` tools (large, fast, deliberately
  "performed" motions), and any fast/instant snapping in face tracking.

## Listening state (`moves.py`, `MovementManager`)

When `set_listening(True)` fires (from transcriber activity):

- Any active `BreathingMove` is stopped immediately and the move queue is
  cleared, so idle head-bob doesn't fight the listening animation
  (`_poll_signals`, `command == "set_listening"`).
- A slow **body-yaw oscillation** (±6° at 0.15 Hz) is added in the main
  control loop (`working_loop`, guarded by `self._is_listening`).
- **Ear/antenna sway** replaces the old frozen antenna position: ±8° at
  0.4 Hz, antiphase between the two antennas (`_calculate_blended_antennas`).
- **Face-tracking is muted to zero**, faded in/out over 0.3s
  (`_face_tracking_mute`, in `_update_face_tracking`) — so the head doesn't
  keep turning to follow faces while listening; the body-yaw/ear-sway carry
  the "I'm listening" cue instead.

This replaced an earlier "head yaw scan" behavior (sweeping the head ±10° at
0.2 Hz while listening, implemented as `_listening_scan_loop` in both
`OpenaiRealtimeHandler` and `BerkyLiveHandler`) — removed for being "too
distracting."

## Face-tracking rotation rate limit

Face tracking (`camera_worker.py` computes the offset; `moves.py`'s
`_update_face_tracking` applies it) used to snap the head instantly to a
newly-detected or fast-moving face — described as "faster rotations and
twists" when the robot hears a voice and turns to find the speaker's face.

Fixed by rate-limiting the rotation components (roll/pitch/yaw only —
translation x/y/z is untouched) in `_update_face_tracking`:
`self._face_tracking_rotation_smoothed` steps toward the (mute-scaled) target
by at most `self._max_face_tracking_rotation_rate * dt` per tick, instead of
jumping straight to the target.

Current rate: **8°/s** (`math.radians(8)`, in `MovementManager.__init__`).
History of this number, in case it needs revisiting:
- Started at 30°/s (first attempt at "not instant").
- Dropped to 8°/s per operator feedback ("much slower").
- Operator also asked to match `sweep_look`'s speed (54°/s = 0.9π rad / 3s,
  since `GotoQueueMove` interpolates head yaw and body_yaw linearly over the
  same duration) — but the 8°/s change was deployed before that request was
  finalized. **If asked to speed this back up, 54°/s was the last explicitly
  discussed reference point**, not 30°/s.

## Profile tool config (`profiles/_berkie_reachy_locked_profile/`)

`tools.txt` / `instructions.txt` only affect the **legacy** `berkie-reachy`
app (`main.py` → `OpenaiRealtimeHandler`) — this is the only code path with
LLM tool-calling wired up. The current `berky-reachy` CLI (`berky_reachy.py`)
and `BerkyLiveHandler` have no tool-calling at all; `dance`/`play_emotion`
can't be triggered from those paths regardless of this config.

Current state:
- `dance`, `stop_dance`, `play_emotion`, `stop_emotion` — **disabled**
  (commented out). These triggered `DanceQueueMove`/`EmotionQueueMove`
  (`dance_emotion_moves.py`), which include large/fast choreographed head
  motions (e.g. `dizzy_spin`, `sharp_side_tilt`, `interwoven_spirals`) —
  exactly the "dancing"/"stark head movement" behavior the operator wants
  gone. `instructions.txt` was updated to explicitly tell the model not to
  use them.
- `sweep_look` — **enabled** (kept after explicit confirmation, despite
  doing a ~162° head+body sweep — the operator considered this distinct from
  "dancing"). Confirmed this is what fires when the model in the legacy
  `berkie-reachy` app (`OpenaiRealtimeHandler`) decides on its own to "look
  toward" a voice it hears — described as the body "whipping around to find
  the face." Slowed from 3.0s transitions (54°/s) to 12.0s transitions
  (~13.5°/s) in `sweep_look.py` — same rough slowdown factor as the
  face-tracking rate limit below, so it reads as a slow, deliberate turn
  instead of a whip.
- `camera`, `do_nothing`, `head_tracking`, `move_head`, `custom_tool` were
  already disabled before this round of changes (untouched).

If asked to also disable `sweep_look`: comment it out in both `tools.txt`
and remove the line inviting its use in `instructions.txt` (this was done
once already this session, then reverted at the operator's request — see
git log on those two files for the exact diff if needed).
