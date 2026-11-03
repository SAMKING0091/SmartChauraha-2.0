"""
Safety validator - hard rules that no optimiser (rule-based or RL) can break.

  1. Emergency preemption always wins.
  2. A direction whose vehicles have waited > MAX_RED_WAIT is served next.
  3. The same direction cannot be extended more than MAX_CONSECUTIVE times
     while another approach has vehicles waiting.
  4. Green time is clipped to [MIN_GREEN, MAX_GREEN].
  5. Yellow and all-red clearance are enforced by the simulator itself
     (Intersection.apply), so they can never be skipped.
"""

from . import config as C
from .green_optimizer import clip_green


def allowed_directions(snap):
    """Directions a controller may pick right now (used for RL action masking)."""
    dirs = list(C.DIRECTIONS)
    others_waiting = any(snap["queue"][d] > 0 for d in dirs if d != snap["active"])
    if (snap["phase"] == "GREEN" and snap["consecutive"] >= C.MAX_CONSECUTIVE
            and others_waiting):
        dirs.remove(snap["active"])
    return dirs


def starving_direction(snap):
    worst, worst_wait = None, C.MAX_RED_WAIT
    for d in C.DIRECTIONS:
        if snap["queue"][d] > 0 and snap["red_wait"][d] > worst_wait:
            worst, worst_wait = d, snap["red_wait"][d]
    return worst


def validate(snap, direction, green):
    """
    Return (direction, green, rule) where rule names the hard rule that
    changed the proposal, or None if the proposal was accepted unchanged.
    """
    if snap.get("emergency"):
        e = snap["emergency"]
        return e, C.MAX_GREEN, None if direction == e else "EMERGENCY_PREEMPTION"

    starving = starving_direction(snap)
    if starving and snap["red_wait"].get(direction, 0) <= C.MAX_RED_WAIT:
        g = max(clip_green(green), C.MIN_GREEN)
        return starving, g, "MAX_RED_WAIT"

    if direction not in allowed_directions(snap):
        others = [d for d in C.DIRECTIONS if d != direction]
        alt = max(others, key=lambda d: (snap["queue"][d], snap["red_wait"][d]))
        return alt, clip_green(green), "MAX_CONSECUTIVE"

    g = clip_green(green)
    return direction, g, None if g == green else "GREEN_LIMITS"
