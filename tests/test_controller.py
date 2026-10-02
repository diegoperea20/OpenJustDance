"""Game controller tests (pure logic)."""

import pytest

from apps.desktop.controller import GameController
from engine.score_engine import PoseComparator, Scorer
from engine.pose_engine.skeleton import Joint, Pose

W1 = {name: 0.05 for name in [
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle"]}


def _pose():
    return Pose(joints={
        "head": Joint(0.0, 0.4, 0.0, 1.0),
        "left_shoulder": Joint(-0.3, 0.3, 0.0, 1.0),
        "right_shoulder": Joint(0.3, 0.3, 0.0, 1.0),
        "left_elbow": Joint(-0.35, 0.2, 0.0, 1.0),
        "right_elbow": Joint(0.35, 0.2, 0.0, 1.0),
        "left_wrist": Joint(-0.3, 0.45, 0.0, 1.0),
        "right_wrist": Joint(0.3, 0.45, 0.0, 1.0),
        "left_hip": Joint(-0.2, 0.0, 0.0, 1.0),
        "right_hip": Joint(0.2, 0.0, 0.0, 1.0),
        "left_knee": Joint(-0.2, -0.3, 0.0, 1.0),
        "right_knee": Joint(0.2, -0.3, 0.0, 1.0),
        "left_ankle": Joint(-0.2, -0.5, 0.0, 1.0),
        "right_ankle": Joint(0.2, -0.5, 0.0, 1.0),
    })


def _controller():
    return GameController(PoseComparator(weights=W1, angle_weight=0.0), Scorer())


def test_perfect_increments_combo_and_score():
    ctrl = _controller()
    pose = _pose()
    r = ctrl.evaluate(pose, pose)
    assert r["rating"] == "Perfect"
    assert r["combo"] == 1
    assert r["score"] == 10
    r = ctrl.evaluate(pose, pose)
    assert r["combo"] == 2
    assert r["score"] == 20


def test_miss_resets_combo():
    ctrl = _controller()
    pose = _pose()
    ctrl.evaluate(pose, pose)
    r = ctrl.evaluate(None, pose)
    assert r["rating"] == "Miss"
    assert r["combo"] == 0
    assert r["score"] == 10  # miss adds no points


def test_neutral_reference_keeps_combo_and_score():
    ctrl = _controller()
    pose = _pose()
    ctrl.evaluate(pose, pose)
    # empty reference = VideoTest with no detection -> neutral, touches no combo/score
    r = ctrl.evaluate(pose, None)
    assert r["rating"] == "Neutral"
    assert r["neutral"] is True
    assert r["combo"] == 1
    assert r["score"] == 10
    # Empty pose is neutral too
    r2 = ctrl.evaluate(pose, Pose())
    assert r2["rating"] == "Neutral"
    assert r2["combo"] == 1


def test_max_combo_and_summary():
    ctrl = _controller()
    pose = _pose()
    for _ in range(5):
        ctrl.evaluate(pose, pose)
    ctrl.evaluate(None, pose)
    for _ in range(3):
        ctrl.evaluate(pose, pose)
    summary = ctrl.summary()
    assert summary["max_combo"] == 5
    assert summary["rating_counts"]["Perfect"] == 8
    assert summary["rating_counts"]["Miss"] == 1
    assert summary["score"] == 80


def test_max_combo_neutral_does_not_break():
    ctrl = _controller()
    pose = _pose()
    for _ in range(5):
        ctrl.evaluate(pose, pose)
    ctrl.evaluate(pose, None)  # neutral -> combo unbroken
    for _ in range(3):
        ctrl.evaluate(pose, pose)
    summary = ctrl.summary()
    assert summary["max_combo"] == 8
    assert summary["frames"] == 8  # neutral counts as no frame
    assert summary["rating_counts"]["Perfect"] == 8
    assert summary["score"] == 80
