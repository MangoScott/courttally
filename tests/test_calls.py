from courttally.calls.inout import call_bounce


def test_clear_in_and_out():
    assert call_bounce((0.0, 5.0)).verdict == "in"
    assert call_bounce((6.0, 5.0)).verdict == "out"
    assert call_bounce((0.0, -13.0)).verdict == "out"


def test_too_close_band_is_honest():
    assert call_bounce((4.10, 0.0), tolerance_m=0.05).verdict == "too_close"
    assert call_bounce((4.13, 0.0), tolerance_m=0.05).verdict == "too_close"
    assert call_bounce((4.10, 0.0), tolerance_m=0.0).verdict == "in"
