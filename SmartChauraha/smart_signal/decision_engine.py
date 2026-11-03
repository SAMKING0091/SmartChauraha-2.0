"""
Decision engine - glues the pieces together.

    snapshot --> controller proposes (direction, green)
             --> safety_validator enforces hard rules
             --> simulator applies it (yellow / all-red inserted there)

Controllers
  RL        Q-learning agent (rl_agent.py)
  PRIORITY  weighted multi-factor score + queue-clearance green (design doc)
  LEGACY    the original SmartChauraha rule: highest density, density-based green
  FIXED     plain round-robin, 20 s each (traditional signal)
"""

from . import config as C
from .green_optimizer import clip_green, queue_clearance_green
from .priority_engine import priority_scores
from .rl_agent import QLearningAgent, action_to_pair
from .safety_validator import allowed_directions, validate

CONTROLLERS = ["RL", "PRIORITY", "LEGACY", "FIXED"]
DIRS = C.DIRECTIONS


# ── individual controllers (pure functions of a snapshot) ───────
def propose_priority(snap):
    scores = priority_scores(snap)
    allowed = allowed_directions(snap)
    d = max(allowed, key=lambda x: scores[x]["score"])
    g = queue_clearance_green(snap["queue"][d], d, snap.get("time_of_day", "NORMAL"))
    return d, g, {"scores": scores}


def propose_legacy(snap):
    density = {d: min(100, round(snap["queue"][d] / C.MAX_QUEUE * 100)) for d in DIRS}
    cur = snap["active"] if snap["active"] in DIRS else DIRS[-1]
    others = sorted((d for d in DIRS if d != cur), key=lambda d: -density[d])
    d = others[0] if density[others[0]] > density[cur] else DIRS[(DIRS.index(cur) + 1) % 4]
    extra = max(0, density[d] - 30) / 70 * (C.MAX_GREEN - C.MIN_GREEN)
    mult = {"PEAK": 1.3, "NORMAL": 1.0, "NIGHT": 0.7}.get(snap.get("time_of_day"), 1.0)
    return d, clip_green((C.MIN_GREEN + extra) * mult), {"density": density}


def propose_fixed(snap):
    cur = snap["active"] if snap["active"] in DIRS else DIRS[-1]
    return DIRS[(DIRS.index(cur) + 1) % 4], 20, {}


class DecisionEngine:
    def __init__(self, controller="RL", agent=None, learn=False):
        self.controller = controller
        self.agent = agent or QLearningAgent()
        self.learn = learn
        self.last = None

    def propose(self, snap):
        if self.controller == "RL":
            a = self.agent.select(snap, allowed_directions(snap))
            if a is None:
                # No experience for this state (or guided exploration while
                # training): fall back to the explainable priority rule.
                d, g, info = propose_priority(snap)
                info["fallback"] = True
                info["q"] = self.agent.explain(snap)
                return d, g, info
            d, g = action_to_pair(a)
            return d, g, {"q": self.agent.explain(snap)}
        if self.controller == "PRIORITY":
            return propose_priority(snap)
        if self.controller == "LEGACY":
            return propose_legacy(snap)
        return propose_fixed(snap)

    def decide(self, sim, apply=True):
        snap = sim.snapshot()
        overridden = sim.override is not None

        if self.controller == "RL":
            if overridden:
                self.agent.reset_episode()       # don't learn across overrides
                sim.delay_acc, sim.starve_acc = 0.0, 0
            else:
                self.agent.observe(sim, learn=self.learn)

        if overridden:
            d, g, info = sim.override["direction"], C.MAX_GREEN, {}
        else:
            d, g, info = self.propose(snap)

        proposed = (d, g)
        d, g, rule = validate(snap, d, g)
        if overridden and sim.override["kind"] == "MANUAL":
            d, rule = sim.override["direction"], "MANUAL_OVERRIDE"

        if self.controller == "RL" and not overridden:
            self.agent.remember(snap, d, g, sim.t)

        decision = {
            "t": sim.t,
            "controller": self.controller,
            "direction": d,
            "green": g,
            "proposed": {"direction": proposed[0], "green": proposed[1]},
            "safety_rule": rule,
            "reason": self.explain(snap, d, g, rule, info),
            "details": info,
        }
        if apply:
            sim.apply(d, g)
        self.last = decision
        return decision

    def explain(self, snap, d, g, rule, info):
        q, w = snap["queue"][d], snap["avg_wait"][d]
        extend = snap["phase"] == "GREEN" and snap["active"] == d
        verb = "Extended" if extend else "Switched to"
        if rule == "EMERGENCY_PREEMPTION" or snap.get("emergency") == d:
            return f"Emergency preemption: {d} held green after safe yellow/all-red."
        if rule == "MANUAL_OVERRIDE":
            return f"Officer override: {d} held green."
        if rule == "MAX_RED_WAIT":
            return f"Safety rule: {d} waited {snap['red_wait'][d]}s (> {C.MAX_RED_WAIT}s), served first."
        if rule == "MAX_CONSECUTIVE":
            return f"Fairness rule: {snap['active']} hit its extension limit; {d} served ({q} queued)."
        base = f"{verb} {d} for {g}s - {q} queued, avg wait {w:.0f}s."
        if self.controller == "RL":
            if info.get("fallback"):
                return base + " RL had no experience of this state - used priority rule as fallback."
            pd = info.get("q", {}).get("per_direction", {})
            if pd and d in pd:
                return base + f" RL: highest Q-value ({pd[d]['q']}) for this traffic state."
            return base + " RL policy."
        if self.controller == "PRIORITY":
            return base + f" Priority score {info['scores'][d]['score']}."
        if self.controller == "LEGACY":
            return base + " Legacy rule: highest density."
        return base + " Fixed round-robin."


def optimize_payload(payload, agent, controller="RL"):
    """
    Stateless helper for POST /api/smart-signal/optimize.
    payload = {
      "queue": {"NORTH": 18, ...},          required
      "avg_wait": {...}, "red_wait": {...}, optional (seconds)
      "emergency": "EAST" | null, "current_green": "NORTH" | null,
      "time_of_day": "NORMAL"
    }
    """
    def per_dir(key):
        src = payload.get(key) or {}
        return {d: float(src.get(d, 0)) for d in DIRS}

    queue = {d: int(v) for d, v in per_dir("queue").items()}
    active = payload.get("current_green")
    snap = {
        "t": 0, "phase": "GREEN" if active in DIRS else "ALL_RED",
        "active": active if active in DIRS else None, "consecutive": 0,
        "queue": queue, "avg_wait": per_dir("avg_wait"), "red_wait": per_dir("red_wait"),
        "time_of_day": payload.get("time_of_day", "NORMAL"),
        "emergency": payload.get("emergency") if payload.get("emergency") in DIRS else None,
    }
    eng = DecisionEngine(controller=controller, agent=agent)
    if snap["emergency"]:
        d, g, info = snap["emergency"], C.MAX_GREEN, {}
    else:
        d, g, info = eng.propose(snap)
    d, g, rule = validate(snap, d, g)
    return {
        "direction": d, "green": g, "controller": controller, "safety_rule": rule,
        "reason": eng.explain(snap, d, g, rule, info),
        "priority_scores": priority_scores(snap),
        "rl": agent.explain(snap),
    }
