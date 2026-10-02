"""Live-vs-reference comparison must be as accurate on horizontal songs
as on vertical ones for the same physical pose.

Reproduces the user case (arm up): projects a synthetic
skeleton onto different framings and runs the EXACT live
CameraThread pipeline (to_compare_space + BodyNormalizer 0.4) against the
extractor path (BodyNormalizer 0.15), with 2px gaussian pixel noise.
"""

import random
import statistics

from engine.normalize_engine.normalizer import BodyNormalizer
from engine.pose_engine.skeleton import Joint, Pose
from engine.score_engine import PoseComparator, compare_space_prescale, to_compare_space

NAMES = ["head", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip",
         "right_hip", "left_knee", "right_knee", "left_ankle",
         "right_ankle"]

BODY_UP = {
    "head": (0.0, 1.62),
    "left_shoulder": (-0.21, 1.42), "right_shoulder": (0.21, 1.42),
    "left_elbow": (-0.26, 1.15), "right_elbow": (0.30, 1.60),
    "left_wrist": (-0.28, 0.90), "right_wrist": (0.33, 1.85),
    "left_hip": (-0.10, 0.95), "right_hip": (0.10, 0.95),
    "left_knee": (-0.11, 0.50), "right_knee": (0.11, 0.50),
    "left_ankle": (-0.12, 0.05), "right_ankle": (0.12, 0.05),
}
BODY_DOWN = dict(BODY_UP, **{"right_elbow": (0.26, 1.15),
                             "right_wrist": (0.28, 0.90)})

LIVE_FRAME = (640, 480, 0.70)
N = 60


def _render(body, W, H, fill_frac, rng, vis_map=None):
    ppm = fill_frac * H / 1.70
    ox, oy = W * 0.5, H * 0.92
    pose = Pose()
    for n in NAMES:
        X, Y = body[n]
        u = ox + X * ppm + rng.gauss(0, 2.0)
        v = oy - Y * ppm + rng.gauss(0, 2.0)
        vv = vis_map.get(n, 1.0) if vis_map else 1.0
        pose.joints[n] = Joint(x=u / W, y=v / H, z=0.0, visibility=vv)
    return pose


def _trial(ref_aspect, ref_frame, live_body, seed, ref_vis_map=None):
    rng = random.Random(seed)
    ref = BodyNormalizer(visibility_threshold=0.15).normalize(
        _render(BODY_UP, *ref_frame, rng=rng, vis_map=ref_vis_map))
    k = compare_space_prescale(ref_aspect, LIVE_FRAME[0] / LIVE_FRAME[1])
    live = BodyNormalizer().normalize(
        to_compare_space(_render(live_body, *LIVE_FRAME, rng=rng), k))
    return PoseComparator().compare(live, ref)


def _mean_err(ref_aspect, ref_frame, live_body=BODY_UP, ref_vis_map=None):
    errs = [_trial(ref_aspect, ref_frame, live_body, 3000 + i,
                   ref_vis_map=ref_vis_map)["error"] for i in range(N)]
    assert all(e is not None for e in errs)
    return statistics.mean(errs)


def test_vertical_same_pose_is_perfect():
    assert _mean_err(720 / 1280, (720, 1280, 0.75)) < 0.05


def test_horizontal_same_pose_is_great_or_better():
    # Same physical pose as vertical: previously ~0.27 (Ok/Miss), now < 0.12.
    assert _mean_err(1280 / 720, (1280, 720, 0.35)) < 0.12


def test_horizontal_mid_visibility_still_tracks():
    low = {"left_wrist": 0.30, "right_wrist": 0.32, "left_elbow": 0.38,
           "right_elbow": 0.35, "left_ankle": 0.28, "right_ankle": 0.31}
    assert _mean_err(1280 / 720, (1280, 720, 0.35), ref_vis_map=low) < 0.12


def test_different_pose_still_miss_despite_alignment():
    errs = [_trial(1280 / 720, (1280, 720, 0.35), BODY_DOWN, 4000 + i)["error"]
            for i in range(20)]
    assert min(errs) > 0.30


def test_align_angle_reported():
    res = _trial(1280 / 720, (1280, 720, 0.35), BODY_UP, 1234)
    assert isinstance(res["align_angle_deg"], float)
