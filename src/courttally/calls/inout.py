"""Line calling: bounce point vs court polygon with an honest confidence."""

from __future__ import annotations

from dataclasses import dataclass

from courttally.court import geometry


@dataclass(frozen=True)
class Call:
    verdict: str  # "in" | "out" | "too_close"
    margin_m: float  # signed distance to nearest line: positive = inside


def margin_to_lines(pt_m: tuple[float, float], singles: bool = True) -> float:
    """Signed distance from the point to the nearest boundary line (positive inside)."""
    x, y = pt_m
    w = geometry.HALF_SINGLES if singles else geometry.HALF_DOUBLES
    return min(w - abs(x), geometry.HALF_LENGTH - abs(y))


def call_bounce(pt_m: tuple[float, float], singles: bool = True, tolerance_m: float = 0.05) -> Call:
    """Call in/out, but say ``too_close`` inside the tolerance band rather than guess."""
    m = margin_to_lines(pt_m, singles)
    if abs(m) <= tolerance_m:
        return Call("too_close", m)
    return Call("in" if m > 0 else "out", m)
