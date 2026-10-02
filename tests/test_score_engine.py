import pytest

from engine.score_engine import PoseComparator, Scorer
from engine.score_engine.comparator import _angle_2d
from engine.pose_engine.skeleton import Joint, Pose

W1 = {"head": 0.05, "left_shoulder": 0.075, "right_shoulder": 0.075, "left_elbow": 0.075,
      "right_elbow": 0.075, "left_wrist": 0.125, "right_wrist": 0.125, "left_hip": 0.05,
      "right_hip": 0.05, "left_knee": 0.075, "right_knee": 0.075, "left_ankle": 0.075,
      "right_ankle": 0.075}


def _pose(wrist_x=0.3, wrist_y=0.45):
    return Pose(
        joints={
            "head": Joint(0.0, 0.4, 0.0, 1.0),
            "left_shoulder": Joint(-0.3, 0.3, 0.0, 1.0),
            "right_shoulder": Joint(0.3, 0.3, 0.0, 1.0),
            "left_elbow": Joint(-0.35, 0.2, 0.0, 1.0),
            "right_elbow": Joint(0.35, 0.2, 0.0, 1.0),
            "left_wrist": Joint(-wrist_x, wrist_y, 0.0, 1.0),
            "right_wrist": Joint(wrist_x, wrist_y, 0.0, 1.0),
            "left_hip": Joint(-0.2, 0.0, 0.0, 1.0),
            "right_hip": Joint(0.2, 0.0, 0.0, 1.0),
            "left_knee": Joint(-0.2, -0.3, 0.0, 1.0),
            "right_knee": Joint(0.2, -0.3, 0.0, 1.0),
            "left_ankle": Joint(-0.2, -0.5, 0.0, 1.0),
            "right_ankle": Joint(0.2, -0.5, 0.0, 1.0),
        }
    )


def test_angle_2d():
    a = Joint(0.0, 0.0)
    b = Joint(0.0, 0.0)
    c = Joint(1.0, 0.0)
    assert _angle_2d(Joint(0.0, 1.0), b, c) == pytest.approx(90.0)


def test_identical_poses_zero_error():
    comparator = PoseComparator(weights=W1, angle_weight=0.0)
    pose = _pose()
    result = comparator.compare(pose, pose)
    assert result["tracking"] is True
    assert result["error"] == pytest.approx(0.0, abs=1e-9)


def test_moved_wrist_increases_error():
    comparator = PoseComparator(weights=W1, angle_weight=0.0)
    ref = _pose()
    player = _pose(wrist_x=0.5, wrist_y=0.5)
    r_ok = comparator.compare(ref, ref)
    comparator.reset()
    r_moved = comparator.compare(player, ref)
    assert r_moved["error"] > r_ok["error"]
    # the wrist weighs 25%, the error must not be zero
    assert r_moved["joint_errors"]["right_wrist"] > 0


def test_weights_respected():
    comparator = PoseComparator(weights=W1, angle_weight=0.0)
    ref = _pose()

    player_head = _pose()
    player_head.joints["head"] = Joint(0.5, 0.4, 0.0, 1.0)  # 0.5 error on head (5% weight)
    comparator.reset()
    r_head = comparator.compare(player_head, ref)

    player_wrist = _pose(wrist_x=0.8, wrist_y=0.45)  # 0.5 error on wrist (12.5% weight)
    comparator.reset()
    r_wrist = comparator.compare(player_wrist, ref)

    assert r_wrist["error"] > r_head["error"]


def test_missing_joint_renormalizes_weights():
    comparator = PoseComparator(weights=W1, angle_weight=0.0)
    ref = _pose()
    player = _pose()
    del player.joints["head"]
    comparator.reset()
    result = comparator.compare(player, ref)
    assert result["tracking"] is True
    assert result["error"] is not None


def test_empty_pose_no_tracking():
    comparator = PoseComparator(weights=W1)
    result = comparator.compare(Pose(), _pose())
    assert result["tracking"] is False
    assert result["error"] is None


def test_angle_terms_increase_error_when_arm_rotated():
    comparator = PoseComparator(weights=W1, angle_weight=1.0)
    ref = _pose()
    player = _pose()
    player.joints["right_elbow"] = Joint(0.5, 0.4, 0.0, 1.0)  # different angle
    result = comparator.compare(player, ref)
    assert result["error"] > 0.0


def test_scorer_ratings():
    scorer = Scorer()
    assert scorer.rate(0.01) == "Perfect"
    assert scorer.rate(0.07) == "Great"
    assert scorer.rate(0.15) == "Good"
    assert scorer.rate(0.25) == "Ok"
    assert scorer.rate(0.5) == "Miss"
    assert scorer.rate(None) == "Miss"


def test_scorer_scores():
    scorer = Scorer()
    assert scorer.score(0.0) == pytest.approx(100.0)
    assert scorer.score(0.5) == pytest.approx(50.0)
    assert scorer.score(None) == 0.0
    assert scorer.score(1.5) == 0.0
