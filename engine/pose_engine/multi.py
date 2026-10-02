"""Unified multi-person pipeline: YOLO-Pose first, HOG+MediaPipe fallback.

Used by the offline extractor (`tools/pose_extractor`) and the live
camera thread (`apps/desktop/threads`). They previously had their own
duplicated, divergent logic (YOLO with a `multi_hits>=3` gate that never
fired when HOG saw 1 person) → 4-person videos like aespavideo.mp4 ended
up with 1 pose. The order is now always:

1. Multi YOLO-Pose (1 forward, up to N people, global 0..1 coords).
2. When YOLO is unavailable or yields <1, sensitive HOG + tracker + per-crop MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.pose_engine.skeleton import Pose

Bbox = Tuple[int, int, int, int]  # x, y, w, h in frame pixels


def _iou(a: Bbox, b: Bbox) -> float:
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


def dedup_boxes(boxes: List[Bbox], iou_thresh: float = 0.30) -> List[Bbox]:
    """Removes same-dancer multi-scale duplicates (largest area first)."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3], reverse=True)
    kept: List[Bbox] = []
    for i in order:
        if all(_iou(boxes[i], k) <= iou_thresh for k in kept):
            kept.append(boxes[i])
    # stable left→right order for slots
    kept.sort(key=lambda b: b[0])
    return kept


class StableSlotMapper:
    """Assigns bboxes to stable 0..N-1 slots (dancer_id) with Hungarian.

    Sorting by x every frame causes ID-switches when dancers cross.
    Here `prev_centers` is kept per slot and assignment uses distance cost,
    with x-order fallback on the first frame.
    """

    def __init__(self, max_slots: int = 4):
        self.max_slots = max_slots
        self.prev_centers: Dict[int, Tuple[float, float]] = {}

    def reset(self) -> None:
        self.prev_centers.clear()

    @staticmethod
    def _center(b: Bbox) -> Tuple[float, float]:
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0)

    def assign(self, boxes: List[Bbox]) -> Dict[int, Bbox]:
        """Returns {slot: bbox} with stable 0..len-1 slots."""
        boxes = list(boxes)[: self.max_slots]
        if not boxes:
            return {}
        if not self.prev_centers:
            boxes_sorted = sorted(boxes, key=lambda b: b[0])
            out = {i: b for i, b in enumerate(boxes_sorted)}
            self.prev_centers = {s: self._center(b) for s, b in out.items()}
            return out
        # Hungarian against previous slots (only slots that still exist).
        prev_slots = sorted(self.prev_centers.keys())[: self.max_slots]
        n_prev, n_new = len(prev_slots), len(boxes)
        try:
            from scipy.optimize import linear_sum_assignment

            cost = np.zeros((n_prev, n_new), dtype=float)
            for i, s in enumerate(prev_slots):
                pcx, pcy = self.prev_centers[s]
                for j, b in enumerate(boxes):
                    cx, cy = self._center(b)
                    diag = max(1.0, float(b[2] + b[3]))
                    cost[i, j] = min(2.0, float(abs(cx - pcx) + abs(cy - pcy)) / diag)
            row, col = linear_sum_assignment(cost)
            used_new = set()
            slot_box: Dict[int, Bbox] = {}
            for r, c in zip(row, col):
                if cost[r, c] > 1.2:
                    continue
                slot_box[prev_slots[r]] = boxes[c]
                used_new.add(c)
            # new (incoming) bboxes → free slots, ordered by x
            free_slots = [s for s in range(self.max_slots) if s not in slot_box]
            remaining = sorted(
                [b for j, b in enumerate(boxes) if j not in used_new],
                key=lambda b: b[0],
            )
            for s, b in zip(free_slots, remaining):
                slot_box[s] = b
            # reindex to 0..k-1 by x for a compact dancer_id, while keeping
            # continuity: when slots were already 0..k-1 they are preserved.
            ordered = sorted(slot_box.items(), key=lambda kv: kv[1][0])
            # When existing slots are already 0..k-1, do not reindex.
            existing = sorted(slot_box.keys())
            if existing == list(range(len(slot_box))):
                self.prev_centers = {s: self._center(b) for s, b in slot_box.items()}
                return dict(slot_box)
            out = {i: b for i, (_, b) in enumerate(ordered)}
            self.prev_centers = {s: self._center(b) for s, b in out.items()}
            return out
        except Exception:
            boxes_sorted = sorted(boxes, key=lambda b: b[0])
            out = {i: b for i, b in enumerate(boxes_sorted)}
            self.prev_centers = {s: self._center(b) for s, b in out.items()}
            return out


@dataclass
class MultiBackendSet:
    """Backend pair for multi: MediaPipe (always) + YOLO (optional)."""

    mp_backend: object = None
    yolo_backend: object = None
    yolo_error: Optional[str] = None


def _pose_snapshot(pose: Pose, min_vis: float = 0.15) -> Dict[str, Tuple[float, float]]:
    """Compact snapshot {joint: (x, y)} with visible joints (global 0..1 coords)."""
    snap: Dict[str, Tuple[float, float]] = {}
    try:
        for name, j in pose.joints.items():
            v = float(getattr(j, "visibility", 1.0))
            if v >= min_vis:
                snap[name] = (float(j.x), float(j.y))
    except Exception:
        pass
    return snap


def _pose_dist(a: Dict[str, Tuple[float, float]], b: Dict[str, Tuple[float, float]]) -> float:
    """Mean distance over shared joints (0..~1). Neutral 0.15 when <3 shared."""
    if not a or not b:
        return 0.15
    common = [k for k in a if k in b]
    if len(common) < 3:
        return 0.15
    import math

    s = 0.0
    for k in common:
        s += math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1])
    return s / max(1, len(common))


def _crop_hist(frame: np.ndarray, bbox: Bbox):
    """Torso HSV histogram (20%-60% vertical, 15% side margin). 8x8, L2."""
    try:
        import cv2

        h_img, w_img = frame.shape[:2]
        x, y, wb, hb = bbox
        x0 = max(0, int(x + wb * 0.15))
        x1 = min(w_img, int(x + wb * 0.85))
        y0 = max(0, int(y + hb * 0.20))
        y1 = min(h_img, int(y + hb * 0.60))
        if x1 <= x0 + 4 or y1 <= y0 + 4:
            return None
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float64)
        n = float((hist ** 2).sum() ** 0.5)
        if n < 1e-9:
            return None
        return hist / n
    except Exception:
        return None


def _hist_cost(a, b) -> float:
    """1 - cosine similarity. Neutral 0.45 when either is missing."""
    if a is None or b is None:
        return 0.45
    try:
        sim = float(np.dot(a, b))
        sim = max(0.0, min(1.0, sim))
        return 1.0 - sim
    except Exception:
        return 0.45


class TrackRegistry:
    """Own dancer tracks with robust anti-occlusion association.

    Trust-but-verify over BoT-SORT/ByteTrack (`model.track(persist=True)`):
    the tracker `track_id` is the primary assignment (direct path) and
    the fused Hungarian only replaces it when it improves the total cost
    with margin. In synchronized dance with similar outfits, pose and
    appearance do not tell people apart and always second-guessing the
    tracker causes swaps; with margin only obvious swaps are fixed
    (teleport / crossed bindings). Each dancer_id keeps its own state:

    * motion: EMA center + velocity -> predicted center per frame.
    * appearance: torso HSV histogram (EMA), robust to similar poses
      with different clothing.
    * pose: joint snapshot (body continuity across frames).

    The dancer<->detection association is Hungarian over fused cost
    (spatial + appearance + pose + size, with bonus on track_id match).
    Dancers without a match but overlapping a matched detection are marked
    OCCLUDED (long TTL, appearance not updated). Expired dancers go to
    a re-identification gallery: a new track_id first tries to
    revive a lost dancer by appearance+pose before taking a free slot.
    """

    # Fused cost weights.
    W_SPATIAL = 1.6
    W_APP = 1.0
    W_POSE = 1.2
    TID_BONUS = 0.35
    # The track_id bonus only counts with small motion: on a large jump the
    # tid was likely swapped in an occlusion and pose+appearance
    # must decide (tracker ID-switch correction).
    TID_SPATIAL_MAX = 0.3
    MATCH_GATE = 1.75  # max cost to match (more = new/lost).
    # Deliberately permissive: after an occlusion the motion prediction goes
    # stale and the spatial gap rises; pose+appearance decide. The real guard
    # against ghosts is the trust-but-verify margin, not this gate.
    SPATIAL_HARD_GATE = 1.4  # impossible predicted/detection distance except re-ID
    OCC_IOU = 0.25  # overlap to declare occlusion
    REID_MAX_COST = 0.62  # combined app+pose threshold to revive a lost one
    # Trust-but-verify: the global Hungarian only replaces the direct
    # track_id assignment when it improves total cost by this margin. Without
    # margin, pose/appearance noise in synchronized dance would permute IDs
    # BoT-SORT+ReID already tracks well. Penalty per unmatched dancer/detection.
    OVERRIDE_MARGIN = 0.3
    MISS_PENALTY = 1.2
    # Rebind: an unmatched tid persisting N frames in a row while a
    # live-occluded dancer claims it by anchor (position+appearance+pose) is
    # re-bound to it. Avoids the eternal "dropped tid with no free slot" when
    # the tracker switches ID on the same person (flip-flop 6->7->6) and the
    # slot stays held by occluded state.
    REBIND_FRAMES = 5
    REBIND_GAP = 0.6
    REBIND_MAX_COST = 0.55
    # The lost gallery retains snapshots far longer than the slot TTL:
    # a dancer covered 5-8s (K-pop formations) must revive with their
    # ID on reappearance instead of taking a fresh slot.
    GALLERY_EXTRA = 150

    def __init__(self, max_slots: int = 4, max_missed: int = 30, occ_ttl: int = 45):
        self.max_slots = max_slots
        # Frames unseen before freeing the slot (dancer truly gone).
        self.max_missed = int(max_missed)
        # Frames an OCCLUDED dancer (covered by another) keeps its slot.
        self.occ_ttl = int(occ_ttl)
        self.track_to_dancer: Dict[int, int] = {}
        self.last_bbox: Dict[int, Bbox] = {}  # dancer_id -> bbox
        self._missed_track: Dict[int, int] = {}  # compat: track_id -> unseen frames
        # Own state per dancer_id.
        self._st: Dict[int, dict] = {}
        # Lost gallery for re-ID: [{dancer, bbox, app, pose, age}].
        self._lost: List[dict] = []
        self._next_neg = -1  # temporary ids for trackless detections
        self.debug_info: dict = {}  # last update: matches/via/events (--debug-track)
        # Streaks of unmatched tids: {tid: [consec_frames, last_bbox]}.
        self._tid_streak: Dict[int, list] = {}

    def reset(self) -> None:
        self.track_to_dancer.clear()
        self.last_bbox.clear()
        self._missed_track.clear()
        self._st.clear()
        self._lost.clear()
        self._next_neg = -1
        self.debug_info = {}
        self._tid_streak.clear()

    @staticmethod
    def _center(b: Bbox) -> Tuple[float, float]:
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0)

    def _nearest_dancer(self, bbox: Bbox) -> Optional[int]:
        if not self.last_bbox:
            return None
        cx, cy = self._center(bbox)
        best, best_d = None, None
        for d, b in self.last_bbox.items():
            px, py = self._center(b)
            diag = max(1.0, float(b[2] + b[3]))
            dist = (abs(cx - px) + abs(cy - py)) / diag
            if best_d is None or dist < best_d:
                best, best_d = d, dist
        if best is not None and best_d is not None and best_d <= 1.0:
            return best
        return None

    # -- costes ---------------------------------------------------------
    def _spatial_gap(self, d: int, bbox: Bbox) -> float:
        """Jump between the dancer's predicted center and the detection (normalized)."""
        last = self._seen_bbox(d)
        px, py = self._predict(d)
        cx, cy = self._center(bbox)
        if last is not None:
            diag = max(1.0, float(last[2] + last[3] + bbox[2] + bbox[3]) / 2.0)
        else:
            diag = max(1.0, float(bbox[2] + bbox[3]))
        return (abs(cx - px) + abs(cy - py)) / diag

    @staticmethod
    def _bbox_gap(a: Bbox, b: Bbox) -> float:
        """Distance between two bbox centers normalized by size."""
        ax, ay = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bx, by = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        diag = max(1.0, float(a[2] + a[3] + b[2] + b[3]) / 2.0)
        return (abs(ax - bx) + abs(ay - by)) / diag

    def _seen_bbox(self, d: int) -> Optional[Bbox]:
        st = self._st.get(d, {})
        b = st.get("_last_seen_bbox", None)
        if b is not None:
            return b
        return self.last_bbox.get(d)

    def _predict(self, d: int) -> Tuple[float, float]:
        st = self._st.get(d)
        b = self._seen_bbox(d)
        if not st or b is None:
            return self._center(b) if b is not None else (0.0, 0.0)
        cx, cy = self._center(b)
        vx, vy = st.get("vel", (0.0, 0.0))
        missed = int(st.get("missed", 0))
        decay = 0.9 ** missed
        return (cx + vx * decay, cy + vy * decay)

    def _match_cost(self, d: int, tid: Optional[int], bbox: Bbox,
                    pose_snap: Dict[str, Tuple[float, float]], app) -> float:
        st = self._st.get(d, {})
        last = self.last_bbox.get(d)
        # 1) spatial vs predicted
        px, py = self._predict(d)
        cx, cy = self._center(bbox)
        if last is not None:
            diag = max(1.0, float(last[2] + last[3] + bbox[2] + bbox[3]) / 2.0)
        else:
            diag = max(1.0, float(bbox[2] + bbox[3]))
        spatial = (abs(cx - px) + abs(cy - py)) / diag
        # 2) torso appearance
        app_c = _hist_cost(st.get("app"), app)
        # 3) pose continuity
        pose_c = _pose_dist(st.get("pose", {}), pose_snap)
        # 4) size change (penalizes jumps to another person/scale)
        import math

        if last is not None:
            a0 = max(1.0, float(last[2] * last[3]))
            a1 = max(1.0, float(bbox[2] * bbox[3]))
            size_p = abs(math.log(a1 / a0)) * 0.3
        else:
            size_p = 0.0
        cost = (self.W_SPATIAL * spatial + self.W_APP * app_c
                + self.W_POSE * pose_c + size_p)
        # bonus: the tracker says it is the same person (anti-jitter hysteresis),
        # but ONLY without a large jump: on a large jump the tid usually
        # arrives swapped from an occlusion and pose+appearance must win.
        if tid is not None and st.get("tid") == tid and spatial <= self.TID_SPATIAL_MAX:
            cost -= self.TID_BONUS
        return cost

    def _solve(self, cost: np.ndarray):
        """Hungarian (scipy) or greedy. Returns matched (i, j) pairs."""
        if cost.size == 0:
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            row, col = linear_sum_assignment(cost)
            return list(zip([int(r) for r in row], [int(c) for c in col]))
        except Exception:
            pairs = []
            used_r, used_c = set(), set()
            flat = sorted([(float(cost[i, j]), i, j)
                           for i in range(cost.shape[0]) for j in range(cost.shape[1])])
            for _, i, j in flat:
                if i not in used_r and j not in used_c:
                    pairs.append((i, j))
                    used_r.add(i)
                    used_c.add(j)
            return pairs

    # -- API ------------------------------------------------------------
    def update(
        self, dets: List[tuple], frame: np.ndarray | None = None,
    ) -> List[Tuple[int, Bbox, Pose]]:
        """Links detections to stable anti-occlusion dancer_ids.

        dets: [(track_id|None, bbox, pose)] or [(track_id|None, bbox, pose, app)].
        frame: current BGR frame (for the torso histogram); optional.
        Returns [(dancer_id, bbox, pose)] sorted by dancer_id.
        """
        # Normalize to (tid, bbox, pose, app|None) + pose snapshot.
        norm: List[tuple] = []
        for t in dets:
            if len(t) == 4:
                tid, bbox, pose, app = t
            else:
                tid, bbox, pose = t[0], t[1], t[2]
                app = None
            try:
                if pose is None or pose.is_empty():
                    continue
            except Exception:
                continue
            norm.append((tid, bbox, pose, app))
        # Cap by area (real people first) when exceeding slots.
        if len(norm) > self.max_slots:
            norm = sorted(norm, key=lambda t: t[1][2] * t[1][3], reverse=True)[:self.max_slots]
        if not norm:
            self._age_all(occluded_only=False)
            self._expire()
            self.debug_info = {"via": "empty", "matches": [], "events": ["no-dets"]}
            return []
        # Current-frame appearance (torso) when not precomputed.
        apps: List = []
        snaps: List[Dict[str, Tuple[float, float]]] = []
        for tid, bbox, pose, app in norm:
            if app is None and frame is not None:
                try:
                    app = _crop_hist(frame, bbox)
                except Exception:
                    app = None
            apps.append(app)
            try:
                snaps.append(_pose_snapshot(pose))
            except Exception:
                snaps.append({})

        dancers = sorted(self._st.keys())
        pre_existing = set(dancers)  # dancers with live state before this frame
        events: List[str] = []
        # Age the lost gallery (long retention: seconds-long occlusions).
        for e in self._lost:
            e["age"] = int(e.get("age", 0)) + 1
        self._lost = [e for e in self._lost if e["age"] <= self.occ_ttl + self.GALLERY_EXTRA][:8]

        matched_d: Dict[int, int] = {}  # dancer -> idx det
        matched_det: Dict[int, int] = {}  # idx det -> dancer
        direct: Dict[int, int] = {}
        chosen: Dict[int, int] = {}
        if dancers:
            cost = np.zeros((len(dancers), len(norm)), dtype=float)
            for i, d in enumerate(dancers):
                for j, (tid, bbox, _pose, _app) in enumerate(norm):
                    cost[i, j] = self._match_cost(d, tid, bbox, snaps[j], apps[j])
            # 1) Direct path: trust the BoT-SORT/ByteTrack track_id
            #    (Ultralytics docs pattern: persist=True + r.boxes.id). Only
            #    discarded on an impossible spatial jump (teleport = recycled
            #    tid or one belonging to another person after overlap).
            direct: Dict[int, int] = {}
            for j, (tid, _bbox, _pose, _app) in enumerate(norm):
                if tid is None:
                    continue
                d = self.track_to_dancer.get(tid)
                if d is None or d not in self._st or d in direct:
                    continue
                if self._spatial_gap(d, _bbox) <= self.SPATIAL_HARD_GATE:
                    direct[d] = j
            # 2) Global path: Hungarian with gates (may fix a real tracker
            #    swap when pose+appearance+motion evidence it).
            global_pairs: Dict[int, int] = {}
            global_cost: Dict[int, float] = {}
            for i, j in self._solve(cost):
                d = dancers[i]
                c = float(cost[i, j])
                st = self._st.get(d, {})
                spat = self._spatial_gap(d, norm[j][1])
                tid_ok = (norm[j][0] is not None and st.get("tid") == norm[j][0]
                          and spat <= self.TID_SPATIAL_MAX)
                if c <= self.MATCH_GATE and (spat <= self.SPATIAL_HARD_GATE or tid_ok):
                    global_pairs[d] = j
                    global_cost[d] = c
            # 3) Trust-but-verify: global only wins when it improves the total
            #    with margin. That way, on normal frames (tracker doing well)
            #    there is no reassignment noise, and an obvious swap is fixed.
            chosen = direct
            if global_pairs != direct:
                pen = self.MISS_PENALTY
                unm_d = [d for d in dancers if d not in direct]
                unm_j = [j for j in range(len(norm)) if j not in set(direct.values())]
                total_direct = (sum(self._match_cost(d, norm[j][0], norm[j][1],
                                                    snaps[j], apps[j])
                                    for d, j in direct.items())
                                + pen * (len(unm_d) + len(unm_j)))
                unm_dg = [d for d in dancers if d not in global_pairs]
                unm_jg = [j for j in range(len(norm)) if j not in set(global_pairs.values())]
                total_global = (sum(global_cost[d] for d in global_pairs)
                                + pen * (len(unm_dg) + len(unm_jg)))
                if total_global + self.OVERRIDE_MARGIN < total_direct:
                    chosen = global_pairs
                    events.append(f"override global {total_global:.2f} < direct {total_direct:.2f}")
            for d, j in chosen.items():
                matched_d[d] = j
                matched_det[j] = d
        via = "global" if (dancers and matched_d and
                           any(matched_d.get(d) != direct.get(d) for d in matched_d)) else "direct"
        overridden = {d for d, j in chosen.items()
                      if d in pre_existing and direct.get(d) != j}
        # Rebind: persistent unmatched tids claimed by live-occluded dancers.
        self._rebind(norm, apps, snaps, matched_d, matched_det, events)
        # Unmatched detections -> lost re-ID or new slot.
        free_new = [j for j in range(len(norm)) if j not in matched_det]
        for j in sorted(free_new, key=lambda k: norm[k][1][0]):
            tid, bbox, pose, app = norm[j]
            revived = self._revive(snaps[j], apps[j], bbox, events)
            if revived is not None:
                matched_d[revived] = j
                matched_det[j] = revived
                continue
            used = set(self._st.keys())
            free = [s for s in range(self.max_slots) if s not in used]
            if not free:
                events.append(f"drop tid{tid} sin slot libre")
                continue
            d = free[0]
            self._st[d] = {"tid": tid, "vel": (0.0, 0.0), "app": None,
                           "pose": {}, "missed": 0, "occ": 0}
            matched_d[d] = j
            matched_det[j] = d
            events.append(f"new tid{tid}->d{d}")
        # Dancers without a match -> occluded (overlapping) or lost.
        matched_boxes = [norm[j][1] for j in matched_det]
        for d in list(self._st.keys()):
            if d in matched_d:
                continue
            last = self.last_bbox.get(d)
            occ = False
            if last is not None:
                px, py = self._predict(d)
                pw, ph = last[2], last[3]
                pred_box = (px - pw / 2.0, py - ph / 2.0, pw, ph)
                for mb in matched_boxes:
                    if _iou((int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])), mb) >= self.OCC_IOU:
                        occ = True
                        break
            st = self._st[d]
            st["missed"] = int(st.get("missed", 0)) + 1
            st["occ"] = int(st.get("occ", 0)) + 1 if occ else 0
            if occ:
                events.append(f"occ d{d} missed={st['missed']}")
            elif st["missed"] == 1:
                events.append(f"lost d{d}")
        # Apply matches: update state + bindings.
        # Anti-poison: matches the override changed vs the direct path
        # update bbox/binding but do NOT rewrite velocity,
        # appearance or pose (they could belong to another person or a
        # 1-frame ghost; they will update when confirmed in later frames).
        out: List[Tuple[int, Bbox, Pose]] = []
        for d, j in matched_d.items():
            tid, bbox, pose, app = norm[j]
            st = self._st[d]
            frozen = d in overridden
            if not frozen:
                last = self.last_bbox.get(d)
                if last is not None:
                    ox, oy = self._center(last)
                    cx, cy = self._center(bbox)
                    pvx, pvy = st.get("vel", (0.0, 0.0))
                    st["vel"] = (0.5 * pvx + 0.5 * (cx - ox), 0.5 * pvy + 0.5 * (cy - oy))
                else:
                    st["vel"] = (0.0, 0.0)
                # Appearance: EMA.
                if app is not None:
                    try:
                        old = st.get("app")
                        if old is not None and old.shape == app.shape:
                            st["app"] = 0.7 * old + 0.3 * app
                        else:
                            st["app"] = app
                    except Exception:
                        st["app"] = app
                st["pose"] = snaps[j]
            elif d in pre_existing:
                events.append(f"frozen d{d} (override no reescribe memoria)")
            st["missed"] = 0
            st["occ"] = 0
            # Identity anchor: dancer's first useful snapshot (for re-ID and
            # rebind; never rewritten with dubious matches).
            if "_anchor_pose" not in st and st.get("pose"):
                st["_anchor_pose"] = dict(st["pose"])
            if "_anchor_app" not in st and st.get("app") is not None:
                st["_anchor_app"] = st["app"]
            if tid is not None:
                old_tid = st.get("tid")
                if old_tid is not None and old_tid != tid:
                    self.track_to_dancer.pop(old_tid, None)
                    self._missed_track.pop(old_tid, None)
                    events.append(f"tid-switch d{d} {old_tid}->{tid}")
                st["tid"] = tid
                self.track_to_dancer[tid] = d
                self._missed_track[tid] = 0
            elif tid is None and st.get("tid") is None:
                st["tid"] = self._next_neg
                self.track_to_dancer[self._next_neg] = d
                self._next_neg -= 1
            self.last_bbox[d] = bbox
            out.append((d, bbox, pose))
        # Compat: age unseen tids.
        seen_tids = {t[0] for t in norm if t[0] is not None}
        for tid in list(self._missed_track.keys()):
            if tid not in seen_tids:
                self._missed_track[tid] = int(self._missed_track.get(tid, 0)) + 1
        self._expire(events)
        out.sort(key=lambda t: t[0])
        # last_bbox: visible ones updated; occluded keep their last bbox
        # (internal state only for prediction, not drawn: absent from out).
        seen = {d: b for d, b, _ in out}
        for d, _b, _p in out:
            self._st[d]["_last_seen_bbox"] = _b
        for d in list(self._st.keys()):
            if d not in seen and "_last_seen_bbox" in self._st[d]:
                seen[d] = self._st[d]["_last_seen_bbox"]
        self.last_bbox = seen
        # Unmatched-tid streaks (for rebind on upcoming frames).
        seen_tids: set = set()
        for j, (tid, bbox, _p, _a) in enumerate(norm):
            if tid is None:
                continue
            seen_tids.add(tid)
            if j in matched_det:
                self._tid_streak.pop(tid, None)
                continue
            prev = self._tid_streak.get(tid)
            if prev is not None:
                gap = self._bbox_gap(prev[1], bbox)
                self._tid_streak[tid] = [int(prev[0]) + 1 if gap <= 1.0 else 1, bbox]
            else:
                self._tid_streak[tid] = [1, bbox]
        for tid in [t for t in self._tid_streak if t not in seen_tids]:
            self._tid_streak.pop(tid, None)
        self.debug_info = {
            "via": via,
            "matches": [(d, norm[j][0], [int(v) for v in norm[j][1]]) for d, j in matched_d.items()],
            "events": list(events),
        }
        return out

    def _rebind(self, norm, apps, snaps, matched_d, matched_det, events) -> None:
        """Re-binds persistently unmatched tids to live-occluded dancers.

        When a tid shows up N frames in a row with no slot while an
        occluded dancer claims it by anchor (position + appearance + pose),
        they are linked. Covers tracker flip-flop (same person, new tid)
        without waiting for the slot to expire.
        """
        cands = [j for j in range(len(norm))
                 if j not in matched_det and norm[j][0] is not None]
        for j in sorted(cands, key=lambda k: norm[k][1][0]):
            tid = norm[j][0]
            streak = self._tid_streak.get(tid)
            if streak is None or int(streak[0]) < self.REBIND_FRAMES:
                continue
            best, best_c = None, None
            for d in self._st.keys():
                if d in matched_d:
                    continue
                st = self._st[d]
                if int(st.get("missed", 0)) < 1:
                    continue
                if self._spatial_gap(d, norm[j][1]) > self.REBIND_GAP:
                    continue
                ac = _hist_cost(st.get("_anchor_app", st.get("app")), apps[j])
                pc = _pose_dist(st.get("_anchor_pose", st.get("pose", {})), snaps[j])
                c = ac + pc
                if best_c is None or c < best_c:
                    best, best_c = d, c
            if best is not None and best_c is not None and best_c <= self.REBIND_MAX_COST:
                bst = self._st[best]
                old = bst.get("tid")
                if old is not None and old != tid:
                    self.track_to_dancer.pop(old, None)
                    self._missed_track.pop(old, None)
                bst["tid"] = tid
                self.track_to_dancer[tid] = best
                self._missed_track[tid] = 0
                matched_d[best] = j
                matched_det[j] = best
                self._tid_streak.pop(tid, None)
                events.append(f"rebind d{best} {old}->{tid} cost={best_c:.2f}")

    def _revive(self, snap, app, bbox, events=None) -> Optional[int]:
        """Re-identifies a lost dancer by appearance+pose. Returns dancer_id or None."""
        if not self._lost:
            return None
        best, best_c = None, None
        for e in self._lost:
            ac = _hist_cost(e.get("anchor_app", e.get("app")), app)
            pc = _pose_dist(e.get("anchor_pose", e.get("pose", {})), snap)
            c = ac + pc
            if best_c is None or c < best_c:
                best, best_c = e, c
        if best is not None and best_c is not None and best_c <= self.REID_MAX_COST:
            d = int(best["dancer"])
            try:
                self._lost.remove(best)
            except Exception:
                pass
            if d not in self._st:
                self._st[d] = {"tid": None, "vel": (0.0, 0.0), "app": best.get("app"),
                               "pose": best.get("pose", {}), "missed": 0, "occ": 0,
                               "_anchor_app": best.get("anchor_app", best.get("app")),
                               "_anchor_pose": dict(best.get("anchor_pose", best.get("pose", {})))}
            if events is not None:
                events.append(f"revive d{d} cost={best_c:.2f}")
            return d
        return None

    def _age_all(self, occluded_only: bool) -> None:
        for st in self._st.values():
            st["missed"] = int(st.get("missed", 0)) + 1

    def _expire(self, events=None) -> None:
        """Frees slots past TTL (occluded hold occ_ttl, the rest max_missed)."""
        for d in list(self._st.keys()):
            st = self._st[d]
            missed = int(st.get("missed", 0))
            occ = int(st.get("occ", 0))
            ttl = self.occ_ttl if occ > 0 else self.max_missed
            # trackless temporaries expire fast
            tid = st.get("tid")
            if isinstance(tid, int) and tid < 0:
                ttl = min(ttl, 5)
            if missed > ttl:
                snap = {"dancer": d, "bbox": self.last_bbox.get(d),
                        "app": st.get("app"), "pose": st.get("pose", {}), "age": 0,
                        "anchor_app": st.get("_anchor_app", st.get("app")),
                        "anchor_pose": dict(st.get("_anchor_pose", st.get("pose", {})))}
                self._lost.append(snap)
                self._lost = self._lost[-8:]
                if tid is not None:
                    self.track_to_dancer.pop(tid, None)
                    self._missed_track.pop(tid, None)
                self._st.pop(d, None)
                self.last_bbox.pop(d, None)
                if events is not None:
                    events.append(f"expired d{d} missed={missed}")
        # compat: drop orphan tids
        for tid in list(self._missed_track.keys()):
            ttl = 5 if (isinstance(tid, int) and tid < 0) else self.max_missed
            if int(self._missed_track.get(tid, 0)) > ttl and tid not in self.track_to_dancer:
                self._missed_track.pop(tid, None)

    def _expire_missed(self) -> None:
        """Compat: alias of the new _expire()."""
        self._expire()


def get_multi_backends(mp_backend=None, yolo: bool = True, max_persons: int = 4,
                       yolo_conf: float = 0.4, yolo_iou: float = 0.6,
                       estimator: str = "yolo",
                       tracker: str | None = "botsort-reid",
                       device: str | None = "auto") -> MultiBackendSet:
    """Creates/factorizes multi backends. YOLO is optional but preferred.

    estimator: "mp" (YOLO boxes+IDs + one MediaPipe per person, 33
    landmarks, YOLO-crop pattern) or "yolo" (direct YOLO keypoints, 17 pts).
    tracker: "botsort-reid" (BoT-SORT + ReID, default anti-overlap),
    "bytetrack"/"bytetrack-dance" (ByteTrack, fast) or stock "botsort".
    """
    from engine.pose_engine import BACKENDS, create_backend

    mp = mp_backend or create_backend("mediapipe-cpu")
    yb = None
    err = None
    if yolo:
        # Prefer YOLO+per-person MediaPipe; fall back to direct YOLO-pose.
        names = ["yolo-mediapipe", "yolo-pose"] if estimator == "mp" else ["yolo-pose"]
        for name in names:
            if name not in BACKENDS:
                continue
            try:
                yb = create_backend(name, max_persons=max_persons, conf=yolo_conf,
                                    iou=yolo_iou, tracker=tracker, device=device)
                break
            except Exception as exc:
                err = str(exc)
                yb = None
    return MultiBackendSet(mp_backend=mp, yolo_backend=yb, yolo_error=err)


def detect_multi_poses(
    frame: np.ndarray,
    backends: MultiBackendSet,
    tracker=None,
    max_persons: int = 4,
    work_width: int = 640,
) -> List[Tuple[Bbox, Pose]]:
    """Detects up to N poses with pose coords in GLOBAL 0..1 space.

    Returns [(bbox, pose_global)] sorted by x. `pose_global` is always
    in full-frame 0..1 coordinates (YOLO already gives them; the
    HOG+crop fallback converts them), so T-pose and drawing agree.
    """
    from engine.pose_engine.detector import detect_persons

    h_img, w_img = frame.shape[:2]
    # 1) YOLO first: 1 forward, N people, no HOG or tracker.
    yb = backends.yolo_backend
    if yb is not None:
        try:
            dets = yb.detect_multi(frame)
            if dets:
                # filter empties and cap, already x-sorted
                clean = [(b, p) for b, p in dets if not p.is_empty()][:max_persons]
                if clean:
                    return clean
        except Exception:
            pass
    # 2) Sensitive HOG fallback + per-crop MediaPipe.
    mp = backends.mp_backend
    if mp is None:
        return []
    try:
        dets = detect_persons(frame, work_width=work_width, sensitive=True)
    except Exception:
        return []
    boxes_all = [(x, y, w, h) for x, y, w, h, _ in dets[:max_persons * 2]]
    boxes_all = dedup_boxes(boxes_all, iou_thresh=0.30)[:max_persons]
    if tracker is not None:
        try:
            tracked = tracker.update(boxes_all)
            if tracked:
                # use tracked bboxes, deduped again in case the
                # tracker keeps duplicates from previous frames
                boxes_all = dedup_boxes(list(tracked.values()), iou_thresh=0.30)[:max_persons]
            elif boxes_all:
                # No active tracks but detections exist: use them directly
                # (the tracker resyncs in the caller when needed).
                pass
            else:
                return []
        except Exception:
            pass
    out: List[Tuple[Bbox, Pose]] = []
    for (x, y, wb, hb) in boxes_all:
        pad_x = int(wb * 0.12)
        pad_y = int(hb * 0.12)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + wb + pad_x)
        y1 = min(h_img, y + hb + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            raw_crop = mp.detect(crop)
        except Exception:
            continue
        if raw_crop.is_empty():
            continue
        # Convert crop 0..1 → global 0..1 so T-pose/drawing/scoring
        # share YOLO's space.
        from engine.pose_engine.skeleton import Joint

        crop_w = float(x1 - x0)
        crop_h = float(y1 - y0)
        global_pose = Pose()
        for name, j in raw_crop.joints.items():
            gx = (x0 + j.x * crop_w) / float(w_img) if w_img else j.x
            gy = (y0 + j.y * crop_h) / float(h_img) if h_img else j.y
            gx = max(0.0, min(1.0, gx))
            gy = max(0.0, min(1.0, gy))
            global_pose.joints[name] = Joint(x=gx, y=gy, z=j.z, visibility=j.visibility)
        out.append(((x0, y0, x1 - x0, y1 - y0), global_pose))
        if len(out) >= max_persons:
            break
    # 3) Mono last resort: full frame (1-person videos where HOG fails).
    if not out:
        try:
            raw = mp.detect(frame)
            if not raw.is_empty():
                out.append(((0, 0, w_img, h_img), raw))
        except Exception:
            pass
    out.sort(key=lambda t: t[0][0])
    return out[:max_persons]


def track_multi_poses(
    frame: np.ndarray,
    backends: MultiBackendSet,
    registry: TrackRegistry,
    max_persons: int = 4,
) -> List[Tuple[int, Bbox, Pose]]:
    """Multi tracking with persistent IDs: returns [(dancer_id, bbox, pose)].

    Uses `YoloPoseBackend.track_multi` (ByteTrack) + `TrackRegistry` with
    its own association (torso motion + appearance + pose) fixing
    ByteTrack ID-switches when one dancer covers another. Poses are in
    global 0..1 coords. When YOLO is unavailable or sees nobody on this
    frame, returns [] (the caller decides on fallback).
    """
    yb = backends.yolo_backend
    if yb is None or registry is None:
        return []
    track_fn = getattr(yb, "track_multi", None)
    if track_fn is None:
        return []
    try:
        dets = track_fn(frame)
    except Exception:
        return []
    if not dets:
        try:
            return registry.update([], frame=frame)[:max_persons]
        except TypeError:
            return registry.update([])[:max_persons]
    triples: List[Tuple[Optional[int], Bbox, Pose]] = [
        (tid, bbox, pose) for tid, bbox, pose in dets if not pose.is_empty()
    ]
    if not triples:
        try:
            return registry.update([], frame=frame)[:max_persons]
        except TypeError:
            return registry.update([])[:max_persons]
    try:
        return registry.update(triples, frame=frame)[:max_persons]
    except TypeError:
        return registry.update(triples)[:max_persons]
