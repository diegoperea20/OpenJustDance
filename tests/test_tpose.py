"""Tolerant T-pose tests (global or crop, MP or YOLO)."""

from engine.pose_engine.detector import is_t_pose, t_pose_score
from engine.pose_engine.skeleton import Joint, Pose


def _t_pose(x0=0.2, y=0.4, shoulder=0.2, arm=0.25, vis=1.0):
    lsx, rsx = x0, x0 + shoulder
    return Pose(joints={
        "left_shoulder": Joint(x=lsx, y=y, z=0.0, visibility=vis),
        "right_shoulder": Joint(x=rsx, y=y, z=0.0, visibility=vis),
        "left_elbow": Joint(x=lsx - arm / 2, y=y, z=0.0, visibility=vis),
        "right_elbow": Joint(x=rsx + arm / 2, y=y, z=0.0, visibility=vis),
        "left_wrist": Joint(x=lsx - arm, y=y, z=0.0, visibility=vis),
        "right_wrist": Joint(x=rsx + arm, y=y, z=0.0, visibility=vis),
        "left_hip": Joint(x=lsx + 0.03, y=y + 0.35, z=0.0, visibility=vis),
        "right_hip": Joint(x=rsx - 0.03, y=y + 0.35, z=0.0, visibility=vis),
    })


def test_t_pose_perfecta_global():
    score, det = t_pose_score(_t_pose())
    assert det["ok"] is True
    assert score >= 0.65
    assert is_t_pose(_t_pose()) is True


def test_t_pose_crop_pequeno():
    # Same gesture but in crop coords (shoulders close in crop 0..1).
    p = _t_pose(x0=0.3, y=0.4, shoulder=0.25, arm=0.22)
    assert is_t_pose(p) is True


def test_t_pose_yolo_conf_baja():
    # YOLO gives confidence ~0.4 instead of visibility 1.0: must still count.
    assert is_t_pose(_t_pose(vis=0.4), vis_thresh=0.3) is True


def test_brazos_caidos_no_es_t():
    p = _t_pose()
    p.joints["left_wrist"] = Joint(x=0.15, y=0.75, z=0.0, visibility=1.0)
    p.joints["right_wrist"] = Joint(x=0.55, y=0.75, z=0.0, visibility=1.0)
    p.joints["left_elbow"] = Joint(x=0.17, y=0.6, z=0.0, visibility=1.0)
    p.joints["right_elbow"] = Joint(x=0.53, y=0.6, z=0.0, visibility=1.0)
    score, det = t_pose_score(p)
    assert is_t_pose(p) is False
    assert score < 0.55


def test_un_brazo_no_es_t():
    p = _t_pose()
    p.joints["left_wrist"] = Joint(x=0.18, y=0.7, z=0.0, visibility=1.0)
    assert is_t_pose(p) is False


def test_pose_vacia_no_es_t():
    assert is_t_pose(Pose()) is False
    s, _ = t_pose_score(Pose())
    assert s == 0.0


def test_tolerancia_vertical():
    # Arms slightly lowered (typical real case) must still count.
    p = _t_pose()
    for n in ("left_wrist", "right_wrist", "left_elbow", "right_elbow"):
        j = p.joints[n]
        p.joints[n] = Joint(x=j.x, y=j.y + 0.05, z=0.0, visibility=1.0)
    assert is_t_pose(p) is True
