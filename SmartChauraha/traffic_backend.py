"""
SmartChauraha - Backend (v3, reinforcement learning)

Flask API around the SmartSignal engine:
  * queue-based intersection simulator (real vehicles, real waiting times)
  * RL signal controller (Deep Q-Network by default) + rule-based baselines
  * hard safety layer: min/max green, yellow + all-red, max red wait,
    fairness limit, safe emergency preemption

Run:
    python train_rl.py        # once (a trained model is already in models/)
    python traffic_backend.py
"""

import csv
import os
import threading
import time

from flask import Flask, jsonify, request

from smart_signal import config as C
from smart_signal.decision_engine import CONTROLLERS, DecisionEngine, optimize_payload
from smart_signal.rl_agent import load_agent
from smart_signal.simulator import Intersection

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

app = Flask(__name__)
try:                                   # flask-cors if installed, else add headers ourselves
    from flask_cors import CORS
    CORS(app)
except ImportError:
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

DIRS = C.DIRECTIONS

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
agent, model_loaded = load_agent(C.RL_MODEL)
if not model_loaded:
    print(f"[!] No trained {C.RL_MODEL} model in {C.MODEL_DIR}/ - run  python train_rl.py\n"
          f"    Until then the RL controller falls back to the priority rule.")
else:
    print(f"[+] Loaded RL model '{C.RL_MODEL}' ({agent.episodes} training episodes)")

lock = threading.Lock()
sim = Intersection(seed=None, time_of_day="NORMAL",
                   initial_queue={"NORTH": 8, "SOUTH": 5, "EAST": 12, "WEST": 3})
sim.metrics_since = 0
engine = DecisionEngine("RL", agent=agent, learn=False)
ui = {"speed": 1, "history": [], "decisions": []}


def load_evaluation():
    path = os.path.join("results", "evaluation.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: (float(v) if k not in ("controller", "key", "time_of_day") else v)
                 for k, v in r.items()} for r in csv.DictReader(f)]


EVALUATION = load_evaluation()


def current_mode():
    if sim.override:
        return sim.override["kind"]
    return "AUTO"


# ─────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────
def step_once():
    if sim.tick() == "DECIDE":
        d = engine.decide(sim)
        ui["decisions"].insert(0, {k: d[k] for k in
                                   ("t", "controller", "direction", "green", "safety_rule", "reason")})
        del ui["decisions"][8:]
    ui["history"].append({"ts": sim.t, **{d: sim.density(d) for d in DIRS}})
    del ui["history"][:-30]


def simulation_loop():
    while True:
        time.sleep(1.0 / ui["speed"])
        with lock:
            step_once()


threading.Thread(target=simulation_loop, daemon=True).start()


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────
@app.route("/api/state")
def get_state():
    with lock:
        snap = sim.snapshot()
        last = engine.last or {}
        rl_view = agent.explain(snap)
        density = {d: sim.density(d) for d in DIRS}
        m = sim.metrics()
        m["window_s"] = sim.t - getattr(sim, "metrics_since", 0)
        return jsonify({
            # fields used by the original UI (kept for compatibility)
            "signals": sim.signals(),
            "density": density,
            "density_label": {d: "LOW" if v < 30 else "MEDIUM" if v < 70 else "HIGH"
                              for d, v in density.items()},
            "vehicle_count": snap["queue"],
            "active_direction": sim.active or "NORTH",
            "phase": "RED_ALL" if sim.phase == "ALL_RED" else sim.phase,
            "timer": max(0, sim.timer),
            "max_timer": sim.max_timer,
            "emergency": {"active": bool(snap["emergency"]), "direction": snap["emergency"]},
            "cycle_count": sim.cycle_count,
            "mode": current_mode(),
            "time_of_day": sim.time_of_day,
            "history": ui["history"][-10:],
            # new in v3
            "queue": snap["queue"],
            "avg_wait": snap["avg_wait"],
            "red_wait": snap["red_wait"],
            "held": sim._held(),
            "controller": engine.controller,
            "controllers": CONTROLLERS,
            "speed": ui["speed"],
            "decision": {k: last.get(k) for k in
                         ("t", "direction", "green", "proposed", "safety_rule", "reason")},
            "decisions": ui["decisions"],
            "priority": (last.get("details") or {}).get("scores"),
            "rl": {
                "model": C.RL_MODEL, "loaded": model_loaded,
                "episodes": agent.episodes, "updates": agent.updates,
                "learning": engine.learn, "epsilon": agent.epsilon,
                "last_reward": agent.last_reward,
                "q_values": rl_view.get("per_direction"),
            },
            "metrics": m,
            "limits": {"min_green": C.MIN_GREEN, "max_green": C.MAX_GREEN,
                       "yellow": C.YELLOW_TIME, "all_red": C.ALL_RED_TIME,
                       "max_red_wait": C.MAX_RED_WAIT},
            "evaluation": EVALUATION,
        })


@app.route("/api/emergency", methods=["POST"])
def set_emergency():
    data = request.json or {}
    direction = data.get("direction", "NORTH")
    with lock:
        if data.get("active") and direction in DIRS:
            sim.preempt(direction, "EMERGENCY")          # safe: yellow + all-red first
        elif sim.override and sim.override["kind"] == "EMERGENCY":
            sim.release()
        snap = sim.snapshot()
    return jsonify({"ok": True, "emergency": {"active": bool(snap["emergency"]),
                                              "direction": snap["emergency"]}})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    mode = (request.json or {}).get("mode", "AUTO")
    with lock:
        if sim.override and sim.override["kind"] == "EMERGENCY":
            pass                                         # emergency must be cleared first
        elif mode == "MANUAL":
            sim.preempt(sim.active or "NORTH", "MANUAL")  # hold current green for the officer
        elif mode == "AUTO":
            sim.release()
    return jsonify({"ok": True, "mode": current_mode()})


@app.route("/api/manual_signal", methods=["POST"])
def manual_signal():
    """Officer override - goes through yellow + all-red, then holds the chosen direction."""
    direction = (request.json or {}).get("direction")
    if direction not in DIRS:
        return jsonify({"error": "Invalid direction"}), 400
    with lock:
        if sim.override and sim.override["kind"] == "EMERGENCY":
            return jsonify({"error": "Clear the emergency first"}), 409
        sim.preempt(direction, "MANUAL")
    return jsonify({"ok": True})


@app.route("/api/time_of_day", methods=["POST"])
def set_time_of_day():
    tod = (request.json or {}).get("time_of_day", "NORMAL")
    with lock:
        if tod in C.ARRIVAL_RATES:
            sim.time_of_day = tod
    return jsonify({"ok": True, "time_of_day": sim.time_of_day})


@app.route("/api/density", methods=["POST"])
def set_density():
    """Slider override: density 0-100 % is converted to a queue of vehicles."""
    data = request.json or {}
    direction = data.get("direction")
    if direction in DIRS:
        with lock:
            sim.set_queue(direction, round(int(data.get("value", 50)) / 100 * C.MAX_QUEUE))
    return jsonify({"ok": True})


@app.route("/api/controller", methods=["POST"])
def set_controller():
    """Switch signal logic: RL | PRIORITY | LEGACY | FIXED. Resets live metrics."""
    ctrl = (request.json or {}).get("controller")
    if ctrl not in CONTROLLERS:
        return jsonify({"error": f"controller must be one of {CONTROLLERS}"}), 400
    with lock:
        engine.controller = ctrl
        agent.reset_episode()
        sim.reset_metrics()
    return jsonify({"ok": True, "controller": ctrl})


@app.route("/api/rl/learning", methods=["POST"])
def set_learning():
    """Turn online learning on/off (agent keeps updating from live traffic)."""
    on = bool((request.json or {}).get("enabled"))
    with lock:
        engine.learn = on
        agent.epsilon = 0.05 if on else C.EPSILON_LIVE
    return jsonify({"ok": True, "learning": on})


@app.route("/api/speed", methods=["POST"])
def set_speed():
    speed = int((request.json or {}).get("speed", 1))
    ui["speed"] = speed if speed in (1, 2, 5, 10) else 1
    return jsonify({"ok": True, "speed": ui["speed"]})


@app.route("/api/metrics/reset", methods=["POST"])
def reset_metrics():
    with lock:
        sim.reset_metrics()
    return jsonify({"ok": True})


@app.route("/api/smart-signal/optimize", methods=["POST"])
def optimize():
    """
    Stateless decision service (design doc section 5).
    Body: {"queue": {...}, "avg_wait": {...}, "red_wait": {...},
           "emergency": "EAST"|null, "current_green": "NORTH"|null,
           "time_of_day": "NORMAL", "controller": "RL"}
    """
    data = request.json or {}
    if not isinstance(data.get("queue"), dict):
        return jsonify({"error": "body must contain a 'queue' object"}), 400
    ctrl = data.get("controller", "RL")
    if ctrl not in CONTROLLERS:
        return jsonify({"error": f"controller must be one of {CONTROLLERS}"}), 400
    with lock:
        return jsonify(optimize_payload(data, agent, ctrl))


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
