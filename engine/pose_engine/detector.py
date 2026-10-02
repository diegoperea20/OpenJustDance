"""Multi-person HOG detector + NMS + T-pose gesture (mediapipe+opencv only)."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from engine.pose_engine.skeleton import Pose


def _hog_detector():
    import cv2
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


_HOG = None


def get_hog():
    global _HOG
    if _HOG is None:
        _HOG = _hog_detector()
    return _HOG


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


def nms(bboxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thresh: float = 0.4) -> List[int]:
    """Return keep indices."""
    if not bboxes:
        return []
    idxs = sorted(range(len(bboxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        remaining = []
        for i in idxs:
            if _iou(bboxes[cur], bboxes[i]) <= iou_thresh:
                remaining.append(i)
        idxs = remaining
    return keep


def detect_persons(frame: np.ndarray, work_width: int = 640, sensitive: bool = False, hit_threshold: float | None = None, nms_thresh: float | None = None) -> List[Tuple[int, int, int, int, float]]:
    """Detect people with HOG. Return a list of (x,y,w,h,score) in original frame coordinates.

    work_width: scaled long side (keeps aspect). Detects at work scale and rescales to original afterwards.
    sensitive: if True uses winStride (4,4) for multi-dance videos (more detections, slower but offline).
    hit_threshold: SVM threshold (None = auto: -0.2 sensitive / 0.0 normal; negative is more sensitive).
    """
    import cv2
    h, w = frame.shape[:2]
    # resize to work size keeping aspect
    if w > work_width or h > work_width:
        scale = work_width / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = w / new_w
        scale_y = h / new_h
    else:
        small = frame
        scale_x = scale_y = 1.0
        new_w, new_h = w, h

    hog = get_hog()
    win_stride = (4, 4) if sensitive else (8, 8)
    # Auto threshold: more permissive sensitive mode for distant dancers (aespavideo 4).
    ht = -0.2 if (hit_threshold is None and sensitive) else (0.0 if hit_threshold is None else hit_threshold)
    try:
        rects, weights = hog.detectMultiScale(small, winStride=win_stride, padding=(8, 8), scale=1.05, hitThreshold=ht)
    except Exception:
        return []
    if len(rects) == 0:
        return []
    # re-scale to original frame
    bboxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    # In sensitive mode smaller/farther people are allowed (width 30 / height 60).
    min_w, min_h = (30, 60) if sensitive else (40, 80)
    for (x, y, wb, hb), sc in zip(rects, weights):
        x0 = int(x * scale_x); y0 = int(y * scale_y)
        wb0 = int(wb * scale_x); hb0 = int(hb * scale_y)
        # minimum size filter: a person cannot be too small
        if wb0 < min_w or hb0 < min_h:
            continue
        bboxes.append((x0, y0, wb0, hb0))
        scores.append(float(sc))
    # NMS: sensitive mode uses a lower threshold to merge multi-scale duplicates of the same
    # dancer (aespavideo: winStride 4,4 duplicates with IoU ~0.34). 0.30 merges them.
    iou_thr = nms_thresh if nms_thresh is not None else (0.30 if sensitive else 0.4)
    keep_idx = nms(bboxes, scores, iou_thresh=iou_thr)
    out: List[Tuple[int, int, int, int, float]] = []
    min_area = 0.008 if sensitive else 0.015
    for i in keep_idx:
        x, y, wb, hb = bboxes[i]
        # filter by relative area: discard tiny detections
        area_ratio = (wb * hb) / float(w * h)
        if area_ratio < min_area:
            continue
        out.append((x, y, wb, hb, scores[i]))
    # sort by x for stable initial assignment
    out.sort(key=lambda b: b[0])
    # max 6 detections to avoid overloading mediapipe
    if len(out) > 6:
        out = out[:6]
    return out


def _joint_angle(a, b, c) -> float:
    """Angle abc in degrees (180 = extended)."""
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosv = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def t_pose_score(pose: Pose, vis_thresh: float = 0.3) -> tuple[float, dict]:
    """Score a T-pose tolerantly, returning (score 0..1, details).

    Works both in global 0..1 image coordinates and in crops
    (uses shoulder width as scale, not absolute pixels), and accepts
    MediaPipe visibility or YOLO confidence (0.3 threshold by default).
    """
    details: dict = {"ok": False, "reasons": []}
    if pose.is_empty():
        details["reasons"].append("empty")
        return 0.0, details
    needed = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    missing = [n for n in needed if not pose.visible(n, vis_thresh)]
    if missing:
        details["reasons"].append(f"oculto:{','.join(missing)}")
        # If 1-2 points are missing but the rest is a clear T, give a partial score instead of 0.
        if len(missing) > 2:
            return 0.0, details
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    le = pose.get("left_elbow")
    re = pose.get("right_elbow")
    lw = pose.get("left_wrist")
    rw = pose.get("right_wrist")
    if None in (ls, rs, le, re, lw, rw):
        details["reasons"].append("sin-joints")
        return 0.0, details
    assert ls is not None and rs is not None and le is not None and re is not None
    assert lw is not None and rw is not None
    shoulder = math.hypot(rs.x - ls.x, rs.y - ls.y)
    if shoulder < 1e-6:
        details["reasons"].append("hombros-juntos")
        return 0.0, details
    # Vertical scale: use torso when hips exist, else shoulders. Tolerates crops.
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    if lh is not None and rh is not None:
        mid_hip_y = (lh.y + rh.y) / 2.0
        mid_sh_y = (ls.y + rs.y) / 2.0
        torso = abs(mid_sh_y - mid_hip_y)
    else:
        torso = shoulder * 1.2
    y_scale = max(torso, shoulder * 0.6, 1e-6)

    def ynorm(dy: float) -> float:
        return abs(dy) / y_scale

    # Sub-scores 0..1 (1 = perfect). Tolerant thresholds.
    # 1) Elbow/wrist height relative to shoulders (<0.35 torso = good, <0.55 acceptable).
    y_vals = [ynorm(lw.y - ls.y), ynorm(rw.y - rs.y), ynorm(le.y - ls.y), ynorm(re.y - rs.y)]
    y_worst = max(y_vals)
    s_y = max(0.0, min(1.0, 1.0 - (y_worst - 0.25) / 0.35))

    # 2) Horizontal extension: wrists outside shoulders + wide span.
    # In image coords x grows rightwards; left_shoulder usually has x < right_shoulder
    # (or the reverse when mirrored). The correct side is used per actual order.
    left_is_left = ls.x <= rs.x
    outer_l, outer_r = (ls, rs) if left_is_left else (rs, ls)
    wrist_l, wrist_r = (lw, rw) if left_is_left else (rw, lw)
    ext_l = (outer_l.x - wrist_l.x) / shoulder  # >0 = outside
    ext_r = (wrist_r.x - outer_r.x) / shoulder
    s_ext = max(0.0, min(1.0, (min(ext_l, ext_r) + 0.1) / 0.5))
    if min(ext_l, ext_r) < -0.15:
        details["reasons"].append("brazos-no-extendidos")
    span = abs(rw.x - lw.x) / max(shoulder, 1e-6)
    s_span = max(0.0, min(1.0, (span - 1.1) / 0.7))

    # 3) Extended arms (elbow angle).
    ang_l = _joint_angle(ls, le, lw)
    ang_r = _joint_angle(rs, re, rw)
    s_ang = max(0.0, min(1.0, (min(ang_l, ang_r) - 120.0) / 30.0))

    # 4) Separated shoulders (avoids false positives with a far frontal torso).
    s_sh = 1.0 if shoulder >= 0.05 else max(0.0, shoulder / 0.05)

    # Height dominates (straight-down arms keep extension/angle but are not a T).
    score = 0.50 * s_y + 0.20 * s_ext + 0.10 * s_span + 0.20 * s_ang
    score *= s_sh
    if missing:
        score *= 0.7  # penalize but not void (1-2 hidden points)
    details.update({
        "s_y": round(s_y, 3), "s_ext": round(s_ext, 3), "s_span": round(s_span, 3),
        "s_ang": round(s_ang, 3), "span": round(span, 3),
        "ang_l": round(ang_l, 1), "ang_r": round(ang_r, 1),
        "y_worst": round(y_worst, 3),
    })
    ok = score >= 0.55 and s_y > 0.3 and s_ang > 0.3 and min(ext_l, ext_r) > -0.15
    details["ok"] = bool(ok)
    if not ok and not details["reasons"]:
        details["reasons"].append(f"score-bajo:{score:.2f}")
    return float(max(0.0, min(1.0, score))), details


def is_t_pose(pose: Pose, vis_thresh: float = 0.3) -> bool:
    """Detect a T-pose: arms horizontal in a cross (tolerant version).

    Accepts global or crop coordinates and MediaPipe visibility or
    YOLO confidence. Threshold: score >= 0.55 (see `t_pose_score`).
    """
    score, details = t_pose_score(pose, vis_thresh=vis_thresh)
    return bool(details.get("ok", False) or score >= 0.65)
