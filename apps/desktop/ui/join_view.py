"""Join lobby with T-pose (multi)."""

from __future__ import annotations

import time
from typing import Dict

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from apps.desktop.i18n import normalize_lang, tr
from apps.desktop.ui.icons import set_button_icon
from apps.desktop.ui.theme import COLORS
from engine.pose_engine.detector import is_t_pose, t_pose_score


class JoinView(QWidget):
    join_finished = Signal(object)  # PlayerManager
    back_requested = Signal()
    # internally needs shared and player_manager
    # Fast, tolerant T-pose: short hold + tolerated dropout so the
    # assignment is nearly instant without false positives from a stray frame.
    TPOSE_HOLD_TIME = 0.6
    TPOSE_DROPOUT_TOL = 0.35
    TPOSE_MIN_SCORE = 0.55

    def __init__(self, shared, player_manager, parent=None) -> None:
        super().__init__(parent)
        self.shared = shared
        self.player_manager = player_manager
        self._lang = "en"
        self._tpose_hold: Dict[int, dict] = {}  # track_id -> {start, last_ok, score}
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._tick)

        self.title_label = QLabel("")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setObjectName("hint")
        self.hint.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setObjectName("video")
        try:
            from apps.desktop.ui.icons import pixmap as _pix

            self._cam_placeholder = _pix("video", 48, COLORS["text2"])
            self.camera_label.setPixmap(self._cam_placeholder)
        except Exception:
            pass

        self.status_label = QLabel("")
        self.status_label.setObjectName("mono")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.players_label = QLabel("")
        self.players_label.setObjectName("subtitle")
        self.players_label.setAlignment(Qt.AlignCenter)

        self.list_label = QLabel("")
        self.list_label.setObjectName("hint")
        self.list_label.setAlignment(Qt.AlignCenter)
        self.list_label.setWordWrap(True)

        self.tpose_label = QLabel("")
        self.tpose_label.setObjectName("hint")
        self.tpose_label.setAlignment(Qt.AlignCenter)

        self.tpose_bar = QProgressBar()
        self.tpose_bar.setRange(0, 100)
        self.tpose_bar.setValue(0)
        self.tpose_bar.setTextVisible(False)
        self.tpose_bar.setFixedHeight(8)
        self.tpose_bar.hide()

        self.add_btn = QPushButton("")
        set_button_icon(self.add_btn, "user-plus", 16)
        self.add_btn.clicked.connect(self._manual_add)

        self.remove_btn = QPushButton("")
        set_button_icon(self.remove_btn, "user-minus", 16)
        self.remove_btn.clicked.connect(self._remove_last)

        self.start_btn = QPushButton("")
        self.start_btn.setObjectName("primary")
        set_button_icon(self.start_btn, "play", 18, COLORS["ink"])
        self.start_btn.clicked.connect(self._start)
        self.start_btn.setEnabled(False)

        self.back_btn = QPushButton("")
        set_button_icon(self.back_btn, "arrow-left", 16)
        self.back_btn.clicked.connect(self.back_requested)

        cam_pane = QWidget()
        cam_layout = QVBoxLayout(cam_pane)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.addWidget(self.camera_label, 1)
        cam_pane.setMinimumWidth(320)

        right = QVBoxLayout()
        right.addWidget(self.players_label)
        right.addWidget(self.list_label)
        right.addWidget(self.tpose_label)
        right.addWidget(self.tpose_bar)
        right.addWidget(self.status_label)
        right.addSpacing(10)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        right.addLayout(btn_row)
        right.addStretch()
        right.addWidget(self.start_btn, 0, Qt.AlignCenter)
        right.addWidget(self.back_btn, 0, Qt.AlignCenter)
        side_pane = QWidget()
        side_pane.setLayout(right)
        side_pane.setMinimumWidth(260)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(cam_pane)
        splitter.addWidget(side_pane)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(self.title_label)
        lay.addWidget(self.hint)
        lay.addWidget(splitter, 1)
        self.retranslate("en")

    def retranslate(self, lang: str | None = None) -> None:
        if lang is not None:
            self._lang = normalize_lang(lang)
        code = self._lang
        self.title_label.setText(tr(code, "join.title"))
        self.hint.setText(tr(code, "join.hint"))
        self.status_label.setText(tr(code, "join.waiting"))
        self.add_btn.setText(tr(code, "join.add"))
        self.add_btn.setToolTip(tr(code, "join.add_tooltip"))
        self.remove_btn.setText(tr(code, "join.remove"))
        self.back_btn.setText(tr(code, "join.back"))
        self._update_labels()

    def begin(self) -> None:
        # Do not clear expected_players here: PlayerCountView sets it. Clear only when missing, for compat.
        exp = getattr(self.player_manager, "expected_players", None)
        if exp is None:
            self.player_manager.reset(clear_expected=True)
        else:
            # reset players but keep the expectation
            self.player_manager.reset(clear_expected=False)
        self._tpose_hold.clear()
        self.tpose_bar.setValue(0)
        self.tpose_bar.hide()
        self._update_labels()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def update_frame(self, qimg) -> None:
        pix = QPixmap.fromImage(qimg).scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.camera_label.setPixmap(pix)

    def update_stats(self, fps: float, brightness: float, backend: str) -> None:
        self.status_label.setText(
            tr(self._lang, "join.camera_status", fps=f"{fps:5.1f}", bright=f"{brightness:3.0f}", backend=backend)
        )

    def _tick(self) -> None:
        # Lobby detection as in calibration: for 1 player use the direct mono pose (far more robust than HOG).
        exp = getattr(self.player_manager, "expected_players", None)
        tracks, _, _ = self.shared.snapshot_multi()
        # Mono fallback: if HOG produced no tracks but there is a legacy pose (as in calibration), synthesize track 1
        if not tracks and exp == 1:
            pose, raw, _, _ = self.shared.snapshot()
            if raw is not None and not raw.is_empty():
                # create a synthetic view for T-pose without mixing up colors
                from apps.desktop.threads import TrackedPerson
                tp = TrackedPerson(track_id=1, bbox=(0, 0, 100, 100), pose=raw, raw=raw, norm_pose=pose)
                tracks = {1: tp}
        active_ids = list(tracks.keys())
        # update player_manager LOST/ACTIVE states
        self.player_manager.update_tracks(active_ids)
        if exp is not None and self.player_manager.num_players() >= exp:
            # no more claims, just refresh labels
            self._update_labels()
            for tid in list(self._tpose_hold.keys()):
                if tid not in tracks:
                    self._tpose_hold.pop(tid, None)
            return
        claimed_any = False
        # For 1 player, show no multiple colors: evaluate track 1 only
        candidates = tracks
        if exp == 1 and len(tracks) > 1:
            # keep only the closest/most central track (lowest ID) to avoid confusion
            first_id = min(tracks.keys())
            candidates = {first_id: tracks[first_id]}
        now = time.monotonic()
        best_progress = 0.0
        best_tid = None
        for tid, tp in candidates.items():
            if self.player_manager.get_by_track(tid) is not None:
                self._tpose_hold.pop(tid, None)
                continue
            # Poses already arrive in global 0..1 coords (YOLO or converted HOG),
            # t_pose_score is scale-invariant via shoulder width.
            pose = tp.raw if tp.raw is not None else tp.pose
            try:
                score, _det = t_pose_score(pose)
            except Exception:
                score = 0.0
            hit = bool(score >= self.TPOSE_MIN_SCORE)
            st = self._tpose_hold.get(tid)
            if hit:
                if st is None:
                    self._tpose_hold[tid] = {"start": now, "last_ok": now, "score": score}
                    st = self._tpose_hold[tid]
                else:
                    st["last_ok"] = now
                    st["score"] = score
                progress = (now - st["start"]) / self.TPOSE_HOLD_TIME
                if progress >= 1.0:
                    # immediate claim
                    p = self.player_manager.claim(tid)
                    if p:
                        self.tpose_label.setText(tr(self._lang, "join.claimed", pid=p.player_id, tid=tid))
                        claimed_any = True
                    self._tpose_hold.pop(tid, None)
                else:
                    if progress > best_progress:
                        best_progress = progress
                        best_tid = tid
            else:
                # short dropout tolerated: no reset if it returns within <0.35s
                if st is not None:
                    if now - st["last_ok"] <= self.TPOSE_DROPOUT_TOL:
                        progress = (now - st["start"]) / self.TPOSE_HOLD_TIME
                        if progress > best_progress:
                            best_progress = progress
                            best_tid = tid
                    else:
                        self._tpose_hold.pop(tid, None)
        if best_tid is not None and not claimed_any:
            pct = int(min(99, best_progress * 100))
            self.tpose_label.setText(tr(self._lang, "join.tpose_hold", tid=best_tid, pct=pct))
            self.tpose_bar.setValue(pct)
            self.tpose_bar.show()
        else:
            self.tpose_bar.hide()
        if claimed_any or True:
            self._update_labels()
        # drop holds for tracks that disappeared (beyond the tolerance)
        for tid in list(self._tpose_hold.keys()):
            if tid not in tracks:
                st = self._tpose_hold.get(tid)
                if st is None or now - st.get("last_ok", 0) > self.TPOSE_DROPOUT_TOL:
                    self._tpose_hold.pop(tid, None)

    @staticmethod
    def _dot(ok: bool) -> str:
        color = COLORS["success"] if ok else COLORS["warn"]
        return f'<span style="color:{color};font-size:16px;">●</span>'

    def _plural(self, n: int) -> str:
        try:
            return "s" if int(n) != 1 else ""
        except Exception:
            return "s"

    def _update_labels(self) -> None:
        n = self.player_manager.num_players()
        exp = getattr(self.player_manager, "expected_players", None)
        code = self._lang
        if exp is not None:
            self.players_label.setText(tr(code, "join.players", n=n, exp=exp))
        else:
            self.players_label.setText(tr(code, "join.players", n=n, exp=4))
        if n == 0:
            if exp is not None:
                self.list_label.setText(tr(code, "join.no_players_exp", exp=exp))
                self.start_btn.setEnabled(False)
                self.start_btn.setText(tr(code, "join.start_wait", exp=exp, plural=self._plural(exp)))
            else:
                self.list_label.setText(tr(code, "join.no_players"))
                self.start_btn.setEnabled(False)
                self.start_btn.setText(tr(code, "join.start_add"))
        else:
            lines = []
            for p in self.player_manager.players:
                ok = p.state in ("ACTIVE", "READY")
                lines.append(f"{self._dot(ok)} " + tr(code, "join.player_line", pid=p.player_id, tid=p.camera_track_id, state=p.state))
            self.list_label.setText("<br>".join(lines))
            # only enable once the expectation is met
            can_start = self.player_manager.is_expected_fulfilled() if exp is not None else (n > 0)
            self.start_btn.setEnabled(bool(can_start))
            if exp is not None:
                if can_start:
                    self.start_btn.setText(tr(code, "join.start_ready", n=n, plural=self._plural(n)))
                else:
                    self.start_btn.setText(tr(code, "join.start_missing", missing=exp - n, plural=self._plural(exp - n), n=n, exp=exp))
            else:
                self.start_btn.setText(tr(code, "join.start_ready", n=n, plural=self._plural(n)))
        self.remove_btn.setEnabled(n > 0)
        # block extra claims once the expectation is reached
        if exp is not None and n >= exp:
            self.tpose_label.setText(tr(code, "join.complete", n=n, exp=exp))
            self.tpose_bar.hide()

    def _manual_add(self) -> None:
        # Do not add once the expectation is reached
        exp = getattr(self.player_manager, "expected_players", None)
        code = self._lang
        if exp is not None and self.player_manager.num_players() >= exp:
            self.tpose_label.setText(tr(code, "join.full_add", exp=exp))
            return
        tracks, _, _ = self.shared.snapshot_multi()
        # Calibration fallback: if there are no multi tracks but a mono pose exists, use track 1
        if not tracks and exp == 1:
            pose, raw, _, _ = self.shared.snapshot()
            if raw is not None and not raw.is_empty():
                p = self.player_manager.claim(1)
                if p:
                    self.tpose_label.setText(tr(code, "join.manual_claim_mono", pid=p.player_id))
                self._update_labels()
                return
            self.tpose_label.setText(tr(code, "join.no_tracks_center"))
            return
        if not tracks:
            self.tpose_label.setText(tr(code, "join.no_tracks"))
            return
        # pick the first unclaimed track
        for tid in sorted(tracks.keys()):
            if self.player_manager.get_by_track(tid) is None:
                p = self.player_manager.claim(tid)
                if p:
                    self.tpose_label.setText(tr(code, "join.manual_claim", pid=p.player_id, tid=tid))
                break
        self._update_labels()

    def _remove_last(self) -> None:
        if not self.player_manager.players:
            return
        last = self.player_manager.players[-1]
        self.player_manager.remove_player(last.player_id)
        self._update_labels()

    def _start(self) -> None:
        if self.player_manager.num_players() == 0:
            return
        exp = getattr(self.player_manager, "expected_players", None)
        if exp is not None and not self.player_manager.is_expected_fulfilled():
            missing = exp - self.player_manager.num_players()
            self.tpose_label.setText(tr(self._lang, "join.missing_start", missing=missing, plural=self._plural(missing)))
            return
        self._timer.stop()
        self.join_finished.emit(self.player_manager)
