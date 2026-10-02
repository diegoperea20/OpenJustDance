"""Simple MultiPersonTracker by IoU + center + Hungarian."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    hits: int = 1
    misses: int = 0
    state: str = "ACTIVE"  # ACTIVE / LOST


class MultiPersonTracker:
    """Light centroid+IoU tracker with Hungarian.

    Does not auto re-identify lost IDs onto another track (PlayerManager owns that).
    """

    def __init__(self, iou_thresh: float = 0.15, max_lost: int = 18, min_hits: int = 1):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """Updates with detections (x,y,w,h). Returns {track_id: bbox} actives.

        Detections must already come with NMS.
        """
        if not detections:
            # bump misses
            to_del = []
            for tid, tr in self.tracks.items():
                tr.misses += 1
                if tr.misses > self.max_lost:
                    to_del.append(tid)
                else:
                    tr.state = "LOST" if tr.misses > 3 else "ACTIVE"
            for tid in to_del:
                del self.tracks[tid]
            return {tid: tr.bbox for tid, tr in self.tracks.items() if tr.state == "ACTIVE"}

        if not self.tracks:
            for det in detections:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = Track(track_id=tid, bbox=det, hits=1, misses=0, state="ACTIVE")
            return {tid: tr.bbox for tid, tr in self.tracks.items()}

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
        n_tr, n_det = len(track_ids), len(detections)

        # cost matrix
        cost = np.zeros((n_tr, n_det), dtype=float)
        for i, tb in enumerate(track_bboxes):
            cx_t, cy_t = _center(tb)
            for j, db in enumerate(detections):
                cx_d, cy_d = _center(db)
                # distance normalized by approximate frame size
                iou = _iou(tb, db)
                dist = math.hypot(cx_t - cx_d, cy_t - cy_d)
                # normalize distance by track diagonal
                diag = math.hypot(tb[2], tb[3]) or 200.0
                dist_norm = min(1.0, dist / (diag * 1.5))
                cost[i, j] = (1.0 - iou) * 0.7 + dist_norm * 0.3
                # very low IoU plus large distance => high cost, avoids matching
                if iou < 0.05 and dist_norm > 0.6:
                    cost[i, j] = 1.5

        # Hungarian
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)
        matched_tr = set()
        matched_det = set()
        # cost threshold for a valid match
        for r, c in zip(row, col):
            if cost[r, c] > 0.75:
                continue
            tid = track_ids[r]
            self.tracks[tid].bbox = detections[c]
            self.tracks[tid].hits += 1
            self.tracks[tid].misses = 0
            self.tracks[tid].state = "ACTIVE"
            matched_tr.add(r)
            matched_det.add(c)

        # unmatched tracks -> bump misses
        for i, tid in enumerate(track_ids):
            if i not in matched_tr:
                tr = self.tracks[tid]
                tr.misses += 1
                if tr.misses > self.max_lost:
                    # removed below
                    pass
                elif tr.misses > 3:
                    tr.state = "LOST"
        # Purge long-lost ones
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].misses > self.max_lost:
                del self.tracks[tid]
        # unmatched detections -> new tracks
        for j, det in enumerate(detections):
            if j not in matched_det:
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = Track(track_id=tid, bbox=det, hits=1, misses=0, state="ACTIVE")
        return {tid: tr.bbox for tid, tr in self.tracks.items() if tr.state == "ACTIVE"}

    def active_ids(self) -> List[int]:
        return [tid for tid, tr in self.tracks.items() if tr.state == "ACTIVE"]

    def get(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)
