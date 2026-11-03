"""
Stage A - multi-factor priority scoring (rule-based baseline).

    P = 0.20 Vn + 0.25 Qn + 0.25 Wn + 0.25 E + 0.05 S

Kept as an explainable baseline that the RL agent is compared against.
"""

from . import config as C


def _norm(x, xmax):
    return min(x / xmax, 1.0) if xmax else 0.0


def priority_scores(snap):
    """Return {direction: {"score": P, "factors": {...}}} for a snapshot."""
    w = C.PRIORITY_WEIGHTS
    out = {}
    for d in C.DIRECTIONS:
        q = snap["queue"][d]
        f = {
            "vehicles": _norm(q, C.V_MAX),          # vehicles waiting on approach
            "queue": _norm(q, C.Q_MAX),
            "waiting": _norm(snap["avg_wait"][d], C.W_MAX),
            "emergency": 1.0 if snap.get("emergency") == d else 0.0,
            "starvation": _norm(snap["red_wait"][d], C.STARVATION_THRESHOLD),
        }
        score = sum(w[k] * f[k] for k in w)
        out[d] = {"score": round(score, 4), "factors": {k: round(v, 3) for k, v in f.items()}}
    return out
