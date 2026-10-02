import pytest

from engine.normalize_engine import BodyNormalizer, PoseSmoother
from engine.pose_engine.skeleton import Joint, Pose


def _standing_pose(shoulder_y=0.3, hip_y=0.5, x_offset=0.1):
    """Simulated standing pose (0..1 image coords)."""
    return Pose(
        joints={
            "head": Joint(x=0.1 + x_offset, y=0.15, z=0.0, visibility=1.0),
            "left_shoulder": Joint(x=-0.15 + x_offset, y=shoulder_y, z=0.0, visibility=1.0),
            "right_shoulder": Joint(x=0.15 + x_offset, y=shoulder_y, z=0.0, visibility=1.0),
            "left_elbow": Joint(x=-0.25 + x_offset, y=0.35, z=0.0, visibility=1.0),
            "right_elbow": Joint(x=0.25 + x_offset, y=0.35, z=0.0, visibility=1.0),
            "left_wrist": Joint(x=-0.3 + x_offset, y=0.45, z=0.0, visibility=1.0),
            "right_wrist": Joint(x=0.3 + x_offset, y=0.45, z=0.0, visibility=1.0),
            "left_hip": Joint(x=-0.1 + x_offset, y=hip_y, z=0.0, visibility=1.0),
            "right_hip": Joint(x=0.1 + x_offset, y=hip_y, z=0.0, visibility=1.0),
            "left_knee": Joint(x=-0.1 + x_offset, y=0.65, z=0.0, visibility=1.0),
            "right_knee": Joint(x=0.1 + x_offset, y=0.65, z=0.0, visibility=1.0),
            "left_ankle": Joint(x=-0.1 + x_offset, y=0.8, z=0.0, visibility=1.0),
            "right_ankle": Joint(x=0.1 + x_offset, y=0.8, z=0.0, visibility=1.0),
        }
    )


def test_normalize_centers_hips_at_origin():
    norm = BodyNormalizer().normalize(_standing_pose())
    left_hip = norm.get("left_hip")
    right_hip = norm.get("right_hip")
    assert (left_hip.x + right_hip.x) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (left_hip.y + right_hip.y) / 2 == pytest.approx(0.0, abs=1e-6)


def test_normalize_scales_shoulder_width_to_1():
    norm = BodyNormalizer().normalize(_standing_pose())
    ls = norm.get("left_shoulder")
    rs = norm.get("right_shoulder")
    width = ((rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2) ** 0.5
    assert width == pytest.approx(1.0, abs=1e-6)


def test_normalize_is_translation_invariant():
    a = BodyNormalizer().normalize(_standing_pose(x_offset=0.0))
    b = BodyNormalizer().normalize(_standing_pose(x_offset=0.6))
    assert a.get("head").x == pytest.approx(b.get("head").x, abs=1e-6)
    assert a.get("left_wrist").y == pytest.approx(b.get("left_wrist").y, abs=1e-6)


def test_normalize_corrects_rotation():
    pose = _standing_pose()
    for name, joint in pose.joints.items():
        # Tilt the body 15 degrees about the hip center.
        cx, cy = 0.1, 0.5
        import math

        angle = math.radians(15)
        dx, dy = joint.x - cx, joint.y - cy
        joint.x = cx + dx * math.cos(angle) - dy * math.sin(angle)
        joint.y = cy + dx * math.sin(angle) + dy * math.cos(angle)

    norm = BodyNormalizer().normalize(pose)
    lh, rh = norm.get("left_hip"), norm.get("right_hip")
    assert lh.y == pytest.approx(rh.y, abs=1e-6)
    assert rh.x > 0 > lh.x


def test_normalize_empty_pose():
    assert BodyNormalizer().normalize(Pose()).is_empty()


def test_smoother_converges():
    smoother = PoseSmoother(alpha=0.5)
    pose = Pose(joints={"head": Joint(x=0.2, y=0.3, z=0.0, visibility=1.0)})
    for _ in range(50):
        out = smoother.smooth(pose)
    assert out.get("head").x == pytest.approx(0.2, abs=1e-6)


def test_smoother_tracks_slow_change():
    smoother = PoseSmoother(alpha=0.4)
    out = smoother.smooth(Pose(joints={"head": Joint(x=0.0, y=0.0, z=0.0, visibility=1.0)}))
    out = smoother.smooth(Pose(joints={"head": Joint(x=1.0, y=1.0, z=0.0, visibility=1.0)}))
    assert 0.0 < out.get("head").x < 1.0
