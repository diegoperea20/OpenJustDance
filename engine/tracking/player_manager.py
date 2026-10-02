"""PlayerManager: TrackID <-> PlayerID <-> DancerID mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apps.desktop.controller import GameController

PlayerState = str  # WAITING, ACTIVE, LOST, READY, FINISHED


@dataclass
class Player:
    player_id: int  # 1..4
    camera_track_id: Optional[int] = None
    reference_dancer_id: Optional[int] = None
    state: PlayerState = "WAITING"
    controller: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "playerId": self.player_id,
            "cameraTrackId": self.camera_track_id,
            "referenceDancerId": self.reference_dancer_id,
            "state": self.state,
        }


class PlayerManager:
    """Manages up to 4 players.

    Rules:
    * Never auto-reassigns a PlayerID to another person (§9). On TrackID loss, marks LOST.
    * claim requires a T-pose and a free TrackID.
    * reference_dancer may repeat (§ allow repeats).
    """

    MAX_PLAYERS = 4

    def __init__(self, comparator_factory=None, scorer_factory=None):
        self.players: List[Player] = []
        self._factory_cmp = comparator_factory
        self._factory_scorer = scorer_factory
        self._next_player_id = 1
        self.expected_players: Optional[int] = None  # set in PlayerCountView (1..4)

    def _make_controller(self):
        from apps.desktop.controller import GameController
        if self._factory_cmp and self._factory_scorer:
            return GameController(self._factory_cmp(), self._factory_scorer())
        from engine.score_engine import PoseComparator, Scorer
        return GameController(PoseComparator(), Scorer())

    def add_player(self, track_id: int) -> Optional[Player]:
        """Tries to create a Player for the track_id. Returns None when impossible."""
        if len(self.players) >= self.MAX_PLAYERS:
            return None
        if any(p.camera_track_id == track_id for p in self.players):
            return None
        pid = self._next_player_id
        self._next_player_id += 1
        p = Player(player_id=pid, camera_track_id=track_id, state="ACTIVE", controller=self._make_controller())
        self.players.append(p)
        return p

    def claim(self, track_id: int) -> Optional[Player]:
        """Claims via T-pose. When a player already exists for that track, no duplicate."""
        if track_id is None:
            return None
        # already claimed?
        for p in self.players:
            if p.camera_track_id == track_id:
                return None
        return self.add_player(track_id)

    def remove_player(self, player_id: int) -> bool:
        for i, p in enumerate(self.players):
            if p.player_id == player_id:
                del self.players[i]
                return True
        return False

    def get_by_player(self, player_id: int) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def get_by_track(self, track_id: int) -> Optional[Player]:
        for p in self.players:
            if p.camera_track_id == track_id:
                return p
        return None

    def assign_dancer(self, player_id: int, dancer_id: int) -> bool:
        """Assigns the reference dancer (repeats allowed)."""
        p = self.get_by_player(player_id)
        if not p:
            return False
        p.reference_dancer_id = int(dancer_id)
        p.state = "READY"
        return True

    def update_tracks(self, active_track_ids: List[int]) -> None:
        """Marks LOST/ACTIVE by whether their camera_track_id is still active. Never reassigns."""
        active_set = set(active_track_ids)
        for p in self.players:
            if p.camera_track_id is None:
                p.state = "WAITING"
            elif p.camera_track_id in active_set:
                if p.state == "LOST":
                    p.state = "ACTIVE"
                elif p.state not in ("READY", "FINISHED"):
                    p.state = "ACTIVE"
            else:
                # temporarily lost -> LOST, keeps camera_track_id
                if p.state not in ("FINISHED", "WAITING"):
                    p.state = "LOST"

    def reconnect(self, player_id: int, new_track_id: int) -> bool:
        """Explicit reconnect §27: the player T-poses again."""
        p = self.get_by_player(player_id)
        if not p:
            return False
        if any(o.camera_track_id == new_track_id for o in self.players if o.player_id != player_id):
            return False
        p.camera_track_id = new_track_id
        p.state = "ACTIVE"
        return True

    def all_ready(self) -> bool:
        if not self.players:
            return False
        return all(p.reference_dancer_id is not None and p.state in ("READY", "ACTIVE") for p in self.players)

    def num_players(self) -> int:
        return len(self.players)

    def set_expected(self, n: int) -> None:
        n = int(n)
        if 1 <= n <= self.MAX_PLAYERS:
            self.expected_players = n
        else:
            self.expected_players = None

    def clear_expected(self) -> None:
        self.expected_players = None

    def is_expected_fulfilled(self) -> bool:
        if self.expected_players is None:
            return self.num_players() > 0
        return self.num_players() == self.expected_players

    def can_add_more(self) -> bool:
        if self.expected_players is not None:
            return len(self.players) < self.expected_players
        return len(self.players) < self.MAX_PLAYERS

    def to_list(self) -> List[dict]:
        return [p.to_dict() for p in self.players]

    def reset(self, clear_expected: bool = False) -> None:
        self.players.clear()
        self._next_player_id = 1
        if clear_expected:
            self.expected_players = None
