"""
Fall detection state machine.

Per tracked pin ID, maintains one of three states:
  STANDING      — initial state when a new ID first appears
  MAYBE_FALLING — fall criterion met for 1-4 consecutive frames
  FALLEN        — criterion sustained for >= SUSTAIN_FRAMES; locked in

Fall criterion (either signal is sufficient — catches all fall orientations):
  - Aspect ratio (h/w) drops below ASPECT_RATIO_THRESHOLD (pin is no longer tall/narrow)
  - OR centroid drops more than CENTROID_DROP_FRACTION * standing_height below its
    standing-state position (catches pins falling toward/away from the camera)

Ball-proximity gate: a pin may only begin transitioning if the ball was detected
within BALL_GATE_RADIUS_FACTOR * pin_height pixels in the last BALL_GATE_FRAMES frames.
If no ball has ever been detected, the gate is skipped (ball detector may have failed).
"""

from typing import Dict, List, Optional, Tuple

ASPECT_RATIO_THRESHOLD = 0.9
CENTROID_DROP_FRACTION = 0.5
SUSTAIN_FRAMES = 5
BALL_GATE_RADIUS_FACTOR = 2.0
BALL_GATE_FRAMES = 60


class _PinState:
    STANDING = "STANDING"
    MAYBE_FALLING = "MAYBE_FALLING"
    FALLEN = "FALLEN"


class _PinRecord:
    def __init__(self) -> None:
        self.state: str = _PinState.STANDING
        self.standing_cy: Optional[float] = None
        self.standing_h: Optional[float] = None
        self.maybe_count: int = 0
        self.fall_frame: Optional[int] = None
        self.fall_timestamp: Optional[float] = None


class FallDetector:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self._pins: Dict[int, _PinRecord] = {}
        self._ball_history: List[Tuple[int, float, float]] = []
        self._last_centroids: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    def update_ball(self, frame_idx: int, cx: float, cy: float) -> None:
        self._ball_history.append((frame_idx, cx, cy))
        cutoff = frame_idx - BALL_GATE_FRAMES
        while self._ball_history and self._ball_history[0][0] < cutoff:
            self._ball_history.pop(0)

    # ------------------------------------------------------------------
    def update_pin(
        self,
        track_id: int,
        frame_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> bool:
        """
        Update state for one tracked pin.
        Returns True only on the frame the pin is first confirmed FALLEN.
        """
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        self._last_centroids[track_id] = (cx, cy)

        if track_id not in self._pins:
            self._pins[track_id] = _PinRecord()

        rec = self._pins[track_id]

        if rec.state == _PinState.FALLEN:
            return False

        if rec.standing_cy is None:
            rec.standing_cy = cy
            rec.standing_h = max(h, 1.0)

        criterion = self._fall_criterion(rec, cy, w, h)
        ball_ok = self._ball_gate(frame_idx, cx, cy, rec.standing_h)

        if criterion and ball_ok:
            if rec.state == _PinState.STANDING:
                rec.state = _PinState.MAYBE_FALLING
                rec.maybe_count = 1
            else:
                rec.maybe_count += 1
                if rec.maybe_count >= SUSTAIN_FRAMES:
                    rec.state = _PinState.FALLEN
                    fall_frame = frame_idx - SUSTAIN_FRAMES + 1
                    rec.fall_frame = fall_frame
                    rec.fall_timestamp = fall_frame / self.fps
                    return True
        else:
            if rec.state == _PinState.MAYBE_FALLING:
                rec.state = _PinState.STANDING
                rec.maybe_count = 0

        return False

    # ------------------------------------------------------------------
    def is_fallen(self, track_id: int) -> bool:
        return track_id in self._pins and self._pins[track_id].state == _PinState.FALLEN

    def get_fall_timestamp(self, track_id: int) -> Optional[float]:
        rec = self._pins.get(track_id)
        return rec.fall_timestamp if rec else None

    def fallen_count(self) -> int:
        return sum(1 for r in self._pins.values() if r.state == _PinState.FALLEN)

    # ------------------------------------------------------------------
    def _fall_criterion(
        self,
        rec: _PinRecord,
        cy: float,
        w: float,
        h: float,
    ) -> bool:
        ar = h / w if w > 0 else 1.0
        aspect_fallen = ar < ASPECT_RATIO_THRESHOLD

        centroid_fallen = False
        if rec.standing_cy is not None and rec.standing_h is not None:
            drop = cy - rec.standing_cy
            centroid_fallen = drop > CENTROID_DROP_FRACTION * rec.standing_h

        return aspect_fallen or centroid_fallen

    def _ball_gate(
        self,
        frame_idx: int,
        pin_cx: float,
        pin_cy: float,
        pin_h: Optional[float],
    ) -> bool:
        if not self._ball_history:
            return True
        radius = BALL_GATE_RADIUS_FACTOR * (pin_h or 50)
        cutoff = frame_idx - BALL_GATE_FRAMES
        for bf, bx, by in reversed(self._ball_history):
            if bf < cutoff:
                break
            dist = ((bx - pin_cx) ** 2 + (by - pin_cy) ** 2) ** 0.5
            if dist <= radius:
                return True
        return False
