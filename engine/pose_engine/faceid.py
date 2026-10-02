"""Mandatory FaceID gallery for multi ingest (trackingdancers/projectv.py style).

BotSort gives frame-to-frame continuity (volatile track_id); the face
periodically corrects to a persistent 1..MAX_DANCERS ID surviving occlusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def cuda_ep_usable() -> bool:
    """Checks that the ONNX CUDA EP really loads (mini session)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import onnxruntime as ort
        from onnx import TensorProto, helper

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception:
        return False


@dataclass
class FaceGallery:
    """Face embedding gallery with a hard MAX_DANCERS cap."""

    max_dancers: int = 4
    match_thresh: float = 0.38
    check_every: int = 6
    face_visible_thresh: float = 0.5
    det_size: tuple = (320, 320)

    gallery: dict = field(default_factory=dict)
    gallery_counts: dict = field(default_factory=dict)
    trackid_to_persistent: dict = field(default_factory=dict)
    last_face_check_frame: dict = field(default_factory=dict)
    next_persistent_id: int = 1
    n_face_calls: int = 0
    _face_app: object = None

    def __post_init__(self):
        self.max_dancers = max(1, min(4, int(self.max_dancers)))
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "FaceID obligatorio pero falta 'insightface'. Instala con: "
                "uv pip install 'insightface>=2.0' 'onnxruntime-gpu==1.22.0' ; "
                "luego verifica con uv run --no-sync python -c \"import insightface\" "
                f"(detalle: {exc})"
            ) from exc
        use_cuda = cuda_ep_usable()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=self.det_size)
        self._face_app = face_app

    # -- gallery ------------------------------------------------------
    def match_or_register(self, embedding):
        best_id, best_sim = None, -1.0
        for pid, emb in self.gallery.items():
            sim = cosine_sim(embedding, emb)
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_id is not None and best_sim >= self.match_thresh:
            n = self.gallery_counts[best_id]
            self.gallery[best_id] = (self.gallery[best_id] * n + embedding) / (n + 1)
            self.gallery_counts[best_id] += 1
            return best_id
        if len(self.gallery) < self.max_dancers:
            pid = self.next_persistent_id
            self.next_persistent_id += 1
            self.gallery[pid] = embedding
            self.gallery_counts[pid] = 1
            return pid
        return best_id

    def resolve_identity(self, track_id, box, frame, frame_idx, face_visible=True):
        is_new = track_id not in self.trackid_to_persistent
        since = frame_idx - self.last_face_check_frame.get(track_id, -10**9)
        if not (is_new or since >= self.check_every):
            return self.trackid_to_persistent[track_id]
        if not face_visible:
            return self.trackid_to_persistent.get(track_id, f"tmp_{track_id}")
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return self.trackid_to_persistent.get(track_id, f"tmp_{track_id}")
        faces = self._face_app.get(crop)
        self.last_face_check_frame[track_id] = frame_idx
        self.n_face_calls += 1
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            pid = self.match_or_register(face.normed_embedding)
            self.trackid_to_persistent[track_id] = pid
            return pid
        return self.trackid_to_persistent.get(track_id, f"tmp_{track_id}")

    def stable_ids(self) -> list:
        return sorted(self.gallery.keys())
