"""
Unit tests for the SmartSignal engine.  Run with either:
    python -m unittest discover tests
    python -m pytest tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_signal import config as C
from smart_signal.decision_engine import DecisionEngine, optimize_payload
from smart_signal.green_optimizer import queue_clearance_green
from smart_signal.priority_engine import priority_scores
from smart_signal.rl_agent import (DQNAgent, LinearQAgent, QLearningAgent, action_to_pair,
                                   encode_state, pair_to_action)
from smart_signal.safety_validator import validate
from smart_signal.simulator import Intersection

DIRS = C.DIRECTIONS


def snap(queue, active=None, phase="ALL_RED", red_wait=None, avg_wait=None,
         emergency=None, consecutive=0):
    return {"t": 0, "phase": phase, "active": active, "consecutive": consecutive,
            "queue": {d: queue.get(d, 0) for d in DIRS},
            "avg_wait": {d: (avg_wait or {}).get(d, 0) for d in DIRS},
            "red_wait": {d: (red_wait or {}).get(d, 0) for d in DIRS},
            "time_of_day": "NORMAL", "emergency": emergency}


def run(sim, engine, seconds, check=None):
    for _ in range(seconds):
        if sim.tick() == "DECIDE":
            engine.decide(sim)
        if check:
            check(sim)


class TestPriority(unittest.TestCase):
    def test_design_doc_example_east_wins(self):
        s = snap({"NORTH": 18, "SOUTH": 9, "EAST": 24, "WEST": 5},
                 avg_wait={"NORTH": 45, "SOUTH": 30, "EAST": 70, "WEST": 20}, emergency="EAST")
        sc = priority_scores(s)
        self.assertEqual(max(sc, key=lambda d: sc[d]["score"]), "EAST")
        self.assertEqual(sc["EAST"]["factors"]["emergency"], 1.0)

    def test_scores_bounded(self):
        s = snap({d: 500 for d in DIRS}, avg_wait={d: 999 for d in DIRS},
                 red_wait={d: 999 for d in DIRS}, emergency="NORTH")
        for d in DIRS:
            self.assertLessEqual(priority_scores(s)[d]["score"], 1.0 + 1e-9)


class TestGreenTime(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(queue_clearance_green(0, "NORTH"), C.MIN_GREEN)
        self.assertEqual(queue_clearance_green(500, "NORTH"), C.MAX_GREEN)

    def test_grows_with_queue(self):
        self.assertLess(queue_clearance_green(5, "EAST"), queue_clearance_green(20, "EAST"))


class TestSafety(unittest.TestCase):
    def test_emergency_overrides_proposal(self):
        d, g, rule = validate(snap({"NORTH": 30}, emergency="WEST"), "NORTH", 20)
        self.assertEqual((d, rule), ("WEST", "EMERGENCY_PREEMPTION"))

    def test_starving_direction_served(self):
        s = snap({"NORTH": 30, "WEST": 2}, red_wait={"WEST": C.MAX_RED_WAIT + 5})
        d, _, rule = validate(s, "NORTH", 30)
        self.assertEqual((d, rule), ("WEST", "MAX_RED_WAIT"))

    def test_consecutive_limit(self):
        s = snap({"NORTH": 30, "EAST": 4}, active="NORTH", phase="GREEN",
                 consecutive=C.MAX_CONSECUTIVE)
        d, _, rule = validate(s, "NORTH", 20)
        self.assertEqual((d, rule), ("EAST", "MAX_CONSECUTIVE"))

    def test_green_clipped(self):
        _, g, _ = validate(snap({"NORTH": 5}), "NORTH", 999)
        self.assertEqual(g, C.MAX_GREEN)


class TestSimulator(unittest.TestCase):
    def test_never_two_greens_and_yellow_always_between(self):
        sim = Intersection(seed=1, time_of_day="PEAK")
        eng = DecisionEngine("PRIORITY")
        prev = {"phase": None, "active": None}

        def check(s):
            sig = s.signals()
            self.assertLessEqual(sum(v != "RED" for v in sig.values()), 1)
            if prev["phase"] == "GREEN" and s.phase == "GREEN":
                self.assertEqual(prev["active"], s.active)   # only extension, no direct swap
            if s.phase == "GREEN" and prev["phase"] not in (None, "GREEN", "ALL_RED"):
                self.fail("green started without all-red clearance")
            prev["phase"], prev["active"] = s.phase, s.active
        run(sim, eng, 1800, check)

    def test_vehicle_conservation(self):
        sim = Intersection(seed=3, time_of_day="NORMAL")
        run(sim, DecisionEngine("FIXED"), 1200)
        m = sim.metrics()
        self.assertEqual(m["arrived"], m["throughput"] + m["in_queue"])

    def test_green_respects_limits(self):
        sim = Intersection(seed=5)
        eng = DecisionEngine("LEGACY")
        greens = []
        for _ in range(1200):
            if sim.tick() == "DECIDE":
                greens.append(eng.decide(sim)["green"])
        self.assertTrue(all(C.MIN_GREEN <= g <= C.MAX_GREEN for g in greens))


class TestEmergency(unittest.TestCase):
    def test_safe_preemption_then_hold(self):
        sim = Intersection(seed=2)
        eng = DecisionEngine("PRIORITY")
        run(sim, eng, 15)
        while not (sim.phase == "GREEN" and sim.active != "WEST"):
            run(sim, eng, 1)
        sim.preempt("WEST")
        phases = []
        for _ in range(12):
            if sim.tick() == "DECIDE":
                eng.decide(sim)
            phases.append(sim.phase)
        self.assertIn("YELLOW", phases)            # never an instant switch
        self.assertIn("ALL_RED", phases)
        self.assertEqual((sim.phase, sim.active), ("GREEN", "WEST"))
        run(sim, eng, 120)                         # held while active
        self.assertEqual((sim.phase, sim.active), ("GREEN", "WEST"))
        sim.release()
        run(sim, eng, 10)
        self.assertIsNone(sim.override)


class TestRL(unittest.TestCase):
    def test_action_mapping_roundtrip(self):
        for a in range(len(DIRS) * len(C.GREEN_CHOICES)):
            self.assertEqual(pair_to_action(*action_to_pair(a)), a)

    def test_state_encoding_is_hashable(self):
        s = encode_state(snap({"NORTH": 50, "EAST": 4}, active="NORTH", phase="GREEN",
                              red_wait={"EAST": 80}))
        self.assertEqual(len(s), 6)
        hash(s)

    def _train_a_little(self, agent):
        eng = DecisionEngine("RL", agent=agent, learn=True)
        agent.epsilon = 0.3
        for seed in range(3):
            agent.reset_episode()
            run(Intersection(seed=seed, time_of_day="PEAK"), eng, 900)
        self.assertGreater(agent.updates + getattr(agent, "steps", 0), 10)

    def test_all_agents_learn_and_save(self):
        for cls in (QLearningAgent, LinearQAgent, DQNAgent):
            with self.subTest(cls.__name__):
                agent = cls(seed=1)
                self._train_a_little(agent)
                with tempfile.TemporaryDirectory() as tmp:
                    p = os.path.join(tmp, "m.json")
                    agent.save(p)
                    other = cls(seed=2)
                    self.assertTrue(other.load(p))

    def test_rl_controller_obeys_safety(self):
        agent = QLearningAgent(seed=0)
        agent.epsilon = 1.0                         # completely random proposals
        sim = Intersection(seed=9, time_of_day="PEAK")
        eng = DecisionEngine("RL", agent=agent)
        for _ in range(1800):
            if sim.tick() == "DECIDE":
                d = eng.decide(sim)
                self.assertTrue(C.MIN_GREEN <= d["green"] <= C.MAX_GREEN)
            self.assertLessEqual(sum(v != "RED" for v in sim.signals().values()), 1)

    def test_optimize_payload_api(self):
        out = optimize_payload({"queue": {"NORTH": 18, "SOUTH": 9, "EAST": 24, "WEST": 5},
                                "avg_wait": {"NORTH": 45, "SOUTH": 30, "EAST": 70, "WEST": 20},
                                "emergency": "EAST"}, QLearningAgent())
        self.assertEqual(out["direction"], "EAST")
        self.assertIn("reason", out)


if __name__ == "__main__":
    unittest.main()
