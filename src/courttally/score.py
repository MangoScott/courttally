"""Tennis scoring state machine. Feed it point winners, read the score at any time.

Supports standard (ad) and no-ad games, 7-point tiebreaks at 6-6, and best-of-N
sets. Players are "A" (near, camera side) and "B" (far).
"""

from __future__ import annotations

from dataclasses import dataclass, field

POINT_NAMES = {0: "0", 1: "15", 2: "30", 3: "40"}


@dataclass
class Score:
    best_of: int = 3
    no_ad: bool = False
    tiebreak_at: int | None = 6
    sets: list[tuple[int, int]] = field(default_factory=list)
    games: list[int] = field(default_factory=lambda: [0, 0])
    points: list[int] = field(default_factory=lambda: [0, 0])
    in_tiebreak: bool = False
    winner: str | None = None
    history: list[dict] = field(default_factory=list)

    def _i(self, player: str) -> int:
        if player not in ("A", "B"):
            raise ValueError(f"player must be 'A' or 'B', got {player!r}")
        return 0 if player == "A" else 1

    @property
    def sets_won(self) -> tuple[int, int]:
        a = sum(1 for s in self.sets if s[0] > s[1])
        return a, len(self.sets) - a

    def _end_game(self, w: int) -> None:
        self.games[w] += 1
        self.points = [0, 0]
        self.in_tiebreak = False
        ga, gb = self.games
        set_over = (max(ga, gb) >= 6 and abs(ga - gb) >= 2) or (
            self.tiebreak_at and max(ga, gb) == self.tiebreak_at + 1
        )
        if set_over:
            self.sets.append((ga, gb))
            self.games = [0, 0]
            sa, sb = self.sets_won
            need = self.best_of // 2 + 1
            if sa == need:
                self.winner = "A"
            elif sb == need:
                self.winner = "B"
        elif self.tiebreak_at and ga == gb == self.tiebreak_at:
            self.in_tiebreak = True

    def point(self, player: str) -> dict:
        """Record a point for ``player``; return the score after it."""
        if self.winner:
            raise ValueError("match is over")
        w = self._i(player)
        o = 1 - w
        self.points[w] += 1
        pw, po = self.points[w], self.points[o]
        if self.in_tiebreak:
            if pw >= 7 and pw - po >= 2:
                self._end_game(w)
        elif self.no_ad:
            if pw >= 4:
                self._end_game(w)
        elif pw >= 4 and pw - po >= 2:
            self._end_game(w)
        state = self.describe()
        state["point_winner"] = player
        self.history.append(state)
        return state

    def game_score(self) -> str:
        a, b = self.points
        if self.in_tiebreak:
            return f"{a}-{b}"
        if a >= 3 and b >= 3:
            if a == b:
                return "Deuce"
            return "Ad A" if a > b else "Ad B"
        return f"{POINT_NAMES.get(a, '40')}-{POINT_NAMES.get(b, '40')}"

    def describe(self) -> dict:
        return {
            "sets": [f"{a}-{b}" for a, b in self.sets],
            "games": f"{self.games[0]}-{self.games[1]}",
            "game": self.game_score(),
            "tiebreak": self.in_tiebreak,
            "winner": self.winner,
        }

    def __str__(self) -> str:
        d = self.describe()
        parts = d["sets"] + [d["games"]]
        return (
            " ".join(parts) + f" ({d['game']})" + (f" winner {d['winner']}" if d["winner"] else "")
        )


def score_points(winners: list[str | None], **kw) -> list[dict]:
    """Score a sequence of point winners; ``None`` entries (untagged) are skipped."""
    s = Score(**kw)
    out = []
    for w in winners:
        if w is None or s.winner:
            out.append(None)
            continue
        out.append(s.point(w))
    return out
