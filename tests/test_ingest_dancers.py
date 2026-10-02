"""Ingest with declared dancer count + webcam router (no real camera)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np


def test_extract_options_num_dancers():
    from pathlib import Path

    from tools.pose_extractor.extract import ExtractOptions

    o = ExtractOptions(video=Path("x.mp4"), num_dancers=4, yolo_model="yolo11m-pose.pt",
                       face_id=True)
    assert o.num_dancers == 4
    assert o.yolo_model == "yolo11m-pose.pt"
    assert o.face_id is True


def test_cli_num_dancers():
    from tools.pose_extractor.cli import build_parser

    args = build_parser().parse_args(["v.mp4", "-o", "out", "--num-dancers", "3"])
    assert args.num_dancers == 3


def test_coco_to_pose_and_iou():
    from tools.pose_extractor.extract import _coco_kpts_to_pose, _iou_xyxy, _kpts_for_track

    kpts = np.zeros((17, 3))
    kpts[0] = [100, 100, 0.9]
    kpts[5] = [80, 120, 0.9]
    kpts[6] = [120, 120, 0.9]
    kpts[11] = [85, 200, 0.9]
    kpts[12] = [115, 200, 0.9]
    pose = _coco_kpts_to_pose(kpts, 200, 200)
    assert not pose.is_empty()
    assert pose.joints["head"].visibility == 0.9
    assert _iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) > 0.99
    assert _iou_xyxy((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    t = [0, 0, 10, 10, 1, 0.9, 0, 0]  # det_ind=0
    kp = _kpts_for_track(t, (0, 0, 10, 10), [(0, 0, 10, 10)], np.stack([kpts]))
    assert kp is not None and kp.shape == (17, 3)


def test_facegallery_missing_gives_helpful_error():
    import importlib.util

    if importlib.util.find_spec("insightface") is not None:
        return  # not applicable with insightface installed
    from engine.pose_engine.faceid import FaceGallery

    try:
        FaceGallery(max_dancers=4)
    except RuntimeError as exc:
        assert "insightface" in str(exc).lower()
        assert "onnxruntime-gpu" in str(exc)
    else:
        raise AssertionError("FaceGallery debió exigir insightface")


def test_camera_router_effective():
    from apps.desktop.settings import Settings
    from apps.desktop.threads import CameraThread, SharedState

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    th = CameraThread(Settings(), SharedState())
    # Camera follows PLAYERS: expected overrides the song.
    th.set_song_num_dancers(4)
    th.set_expected_players(1)
    assert th.effective_dancers() == 1  # 1 player -> at most 1 person on webcam
    th.set_expected_players(3)
    assert th.effective_dancers() == 3
    th.set_expected_players(None)
    th.set_song_num_dancers(2)
    assert th.effective_dancers() == 2  # fallback to the song


def test_import_dialog_num_dancers():
    from pathlib import Path

    from apps.desktop.ui.import_dialog import ImportDialog

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    dlg = ImportDialog(Path("vid.mp4"), Path("songs"))
    dlg.num_dancers_spin.setValue(4)
    assert dlg.get_data()["num_dancers"] == 4
