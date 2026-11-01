"""Seat locking (spec §6.3): a track owns a seat after its anchor stays in that seat for N frames.
If the tracker swaps IDs, the new ID inherits the seat — the seat is the identity, track IDs are
internal only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from edge.calibration.seatmap import SeatMap, person_anchor
from edge.perception.pose_tracker import Person


@dataclass
class SeatState:
    label: str
    track_id: int | None = None
    person: Person | None = None      # this frame's detection, None if not seen
    last_seen: int = -1
    track_history: list[int] = field(default_factory=list)  # every track ID that owned this seat


class SeatLocker:
    def __init__(self, seatmap: SeatMap, lock_frames: int = 8, stale_frames: int = 8):
        """lock_frames: consecutive frames in a seat before a track claims it.
        stale_frames: frames an owner may be missing before another track can take the seat."""
        self.seatmap = seatmap
        self.lock_frames, self.stale_frames = lock_frames, stale_frames
        self.move_frames = 4 * lock_frames  # hysteresis: changing seats needs a much longer dwell
        self.seats = {s.label: SeatState(s.label) for s in seatmap.seats}
        self.track_seat: dict[int, str] = {}
        self.candidate: dict[int, tuple[str | None, int]] = {}
        self.frame = -1
        self.handovers = 0   # tracker ID switches absorbed by the seat

    def update(self, people: list[Person]) -> dict[str, SeatState]:
        self.frame += 1
        for st in self.seats.values():
            st.person = None

        for p in people:
            if p.track_id is None:
                continue
            seat = self.seatmap.locate(person_anchor(p.kpts, p.kpt_conf, p.box))
            label = seat.label if seat else None
            prev, n = self.candidate.get(p.track_id, (None, 0))
            n = n + 1 if label == prev else 1
            self.candidate[p.track_id] = (label, n)

            owned = self.track_seat.get(p.track_id)
            if owned and (label in (owned, None) or n < self.move_frames):
                self._see(owned, p)          # stay put through jitter / leaning over a seat boundary
                continue
            if label is None or n < self.lock_frames:
                continue
            st = self.seats[label]
            if st.track_id not in (None, p.track_id) and self.frame - st.last_seen < self.stale_frames:
                continue                     # seat held by someone still present
            if owned:                        # track genuinely moved seats
                self.seats[owned].track_id = None
            if st.track_id is not None and st.track_id != p.track_id:
                self.track_seat.pop(st.track_id, None)
                self.handovers += 1
            st.track_id = p.track_id
            st.track_history.append(p.track_id)
            self.track_seat[p.track_id] = label
            self._see(label, p)
        return self.seats

    def _see(self, label: str, p: Person):
        st = self.seats[label]
        st.person, st.last_seen = p, self.frame

    @property
    def occupied(self) -> set[str]:
        return {l for l, st in self.seats.items() if st.person is not None}
