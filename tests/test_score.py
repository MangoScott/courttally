import pytest

from courttally.score import Score, score_points


def test_game_points_and_names():
    s = Score()
    assert s.game_score() == "0-0"
    s.point("A")
    assert s.game_score() == "15-0"
    s.point("A")
    s.point("B")
    assert s.game_score() == "30-15"
    s.point("A")
    s.point("A")
    assert s.games == [1, 0] and s.game_score() == "0-0"


def test_deuce_and_advantage():
    s = Score()
    for w in "ABABAB":
        s.point(w)
    assert s.game_score() == "Deuce"
    s.point("A")
    assert s.game_score() == "Ad A"
    s.point("B")
    assert s.game_score() == "Deuce"
    s.point("B")
    s.point("B")
    assert s.games == [0, 1]


def test_no_ad_game_ends_at_four():
    s = Score(no_ad=True)
    for w in "ABABABA":
        s.point(w)
    assert s.games == [1, 0]


def _win_game(s: Score, who: str) -> None:
    for _ in range(4):
        s.point(who)


def test_tiebreak_and_set():
    s = Score()
    for _ in range(6):
        _win_game(s, "A")
        _win_game(s, "B")
    assert s.games == [6, 6] and s.in_tiebreak
    for w in "ABABABABABAB":  # 6-6 in the breaker
        s.point(w)
    assert s.game_score() == "6-6"
    s.point("A")
    s.point("A")
    assert s.sets == [(7, 6)] and s.games == [0, 0] and not s.in_tiebreak


def test_best_of_three_match():
    s = Score()
    for _ in range(2):
        for _ in range(6):
            _win_game(s, "B")
    assert s.sets == [(0, 6), (0, 6)] and s.winner == "B"
    with pytest.raises(ValueError):
        s.point("A")
    assert "winner B" in str(s)


def test_score_points_skips_untagged():
    out = score_points(["A", None, "A"])
    assert out[1] is None and out[2]["game"] == "30-0"
