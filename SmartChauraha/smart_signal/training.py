"""Training and evaluation loops shared by train_rl.py, evaluate.py and the backend."""

import random
import time

from . import config as C
from .decision_engine import DecisionEngine
from .rl_agent import QLearningAgent
from .simulator import Intersection

TOD_MIX = [("PEAK", 0.5), ("NORMAL", 0.3), ("NIGHT", 0.2)]   # peak-heavy: hardest regime


def run_episode(engine, seed, time_of_day, seconds=1800, initial_queue=None):
    sim = Intersection(seed=seed, time_of_day=time_of_day, initial_queue=initial_queue)
    if engine.controller == "RL":
        engine.agent.reset_episode()
    for _ in range(seconds):
        if sim.tick() == "DECIDE":
            engine.decide(sim)
    if engine.controller == "RL" and engine.learn:
        engine.agent.observe(sim, learn=True)    # close the final transition
    return sim.metrics()


def train(episodes=3000, seconds=3600, seed=7, agent=None, log_every=100, progress=None):
    rng = random.Random(seed)
    agent = agent or QLearningAgent(seed=seed)
    engine = DecisionEngine("RL", agent=agent, learn=True)
    history = []
    window = []
    t0 = time.time()
    for ep in range(episodes):
        frac = min(1.0, ep / max(1, 0.8 * episodes))
        agent.epsilon = C.EPSILON_START + frac * (C.EPSILON_END - C.EPSILON_START)
        agent.guide = C.GUIDE_START * (1.0 - frac)
        tod = rng.choices([t for t, _ in TOD_MIX], [w for _, w in TOD_MIX])[0]
        init = {d: rng.randint(0, 15) for d in C.DIRECTIONS}
        m = run_episode(engine, rng.randrange(10**9), tod, seconds, init)
        agent.episodes += 1
        window.append(m["avg_wait"])
        if (ep + 1) % log_every == 0:
            row = {"episode": ep + 1, "epsilon": round(agent.epsilon, 3),
                   "avg_wait": round(sum(window) / len(window), 2),
                   "states": len(getattr(agent, "q", {})), "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            window = []
            if progress:
                progress(row)
    agent.epsilon = C.EPSILON_LIVE
    agent.guide = 0.0
    return agent, history


def evaluate(agent, seeds=range(1000, 1010), seconds=3600, controllers=("FIXED", "LEGACY", "PRIORITY", "RL")):
    """Same seeds (= identical arrival streams) for every controller."""
    results = {}
    for ctrl in controllers:
        for tod in ("PEAK", "NORMAL", "NIGHT"):
            rows = []
            for s in seeds:
                agent.epsilon, agent.guide = 0.0, 0.0
                eng = DecisionEngine(ctrl, agent=agent, learn=False)
                rows.append(run_episode(eng, s, tod, seconds))
            n = len(rows)
            results.setdefault(ctrl, {})[tod] = {
                "avg_wait": round(sum(r["avg_wait"] for r in rows) / n, 2),
                "max_wait": max(r["max_wait"] for r in rows),
                "throughput": round(sum(r["throughput"] for r in rows) / n, 1),
                "left_in_queue": round(sum(r["in_queue"] for r in rows) / n, 1),
            }
    return results
