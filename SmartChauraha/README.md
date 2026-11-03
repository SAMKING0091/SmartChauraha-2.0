# SmartChauraha 3.0 — RL-Optimised Traffic Signal Control

A full-stack simulation of a 4-way intersection whose signals are controlled by a
**reinforcement-learning agent (Deep Q-Network)**, wrapped in a hard safety layer
and compared against rule-based controllers on identical traffic.

---

## What changed from v2

| | v2 (original) | v3 (this version) |
|---|---|---|
| Traffic model | Random "density" number per lane | Real vehicle queues: Poisson arrivals, 1 veh/s discharge on green, every vehicle's waiting time tracked |
| Signal logic | Highest density goes next; green = f(density) × time-of-day multiplier | **RL agent** chooses direction *and* green length (10/20/30 s); can extend a green |
| Explainability | Generic insight text | Every decision has a reason; Q-values per direction shown live; decision log |
| Safety | Emergency switched lights **instantly** | Emergency and police override go through yellow → all-red first; max-red-wait, fairness limit and green limits enforced on every decision |
| Evaluation | None | `evaluate.py` runs every controller on the same traffic and reports average/max wait |
| Design-doc API | — | `POST /api/smart-signal/optimize` (stateless decision service) |

---

## Results

Average wait per vehicle in seconds, lower is better. Each cell is 20 different one-hour
traffic streams; every controller sees **exactly the same vehicles**. These seeds were held
out — never used while designing or tuning the agent.

| Controller | Peak | Normal | Night | Worst single wait (peak) |
|---|---|---|---|---|
| Fixed-time (20 s round robin) | 168.2 | 46.5 | 34.6 | 399 |
| Original SmartChauraha rule | 55.5 | 26.8 | **18.6** | 310 |
| Weighted priority + clearance (design doc) | 57.6 | 27.1 | 19.9 | 397 |
| RL – tabular Q-learning | 59.0 | 33.5 | 26.7 | 508 |
| RL – linear Q-learning | 47.8 | 27.6 | 21.6 | 189 |
| **RL – Deep Q-Network (default)** | **44.8** | **25.6** | 18.9 | 244 |

**Reading the table honestly**

- At **peak**, DQN cuts average waiting by **19 %** versus the original rule and **73 %** versus a fixed-time signal. This is where RL matters: near capacity, the trade-off between serving long queues and losing 5 s of yellow + all-red on every switch is hard to hand-tune.
- At **normal** traffic DQN is best by a small margin; at **night** it effectively ties the original rule (18.9 vs 18.6 s). Light traffic is easy, so there is little left to learn.
- **Tabular Q-learning was not good enough** — with ~32,000 possible states it cannot generalise between similar traffic situations. This is the motivation for function approximation (linear, then DQN).
- Results come from a simulator, not a real junction. Arrival rates are assumptions in `config.py`.

Re-create with `python evaluate.py` → `results/evaluation.md`, `.csv`, `.png`.

---

## Quick start

```bash
pip install -r requirements.txt     # flask, numpy (flask-cors, matplotlib optional)

python traffic_backend.py           # trained model already in models/
# open traffic_frontend.html in Chrome / Firefox / Edge
```

Optional:

```bash
python train_rl.py                  # retrain DQN (3000 simulated hours, ~10 min)
python train_rl.py --model linear   # or: tabular
python evaluate.py                  # compare all controllers
python -m unittest discover tests   # 17 tests
```

---

## How the RL agent works

### The problem as an MDP
Every time a green ends, the controller makes one decision.

| | Definition |
|---|---|
| **State** | For each direction: queue length, average wait, time since last green, whether it is the current green; plus totals and the priority-rule score (19 features, see `FEATURE_NAMES`) |
| **Action** | (direction to serve, green duration ∈ {10, 20, 30} s) → 12 actions. Choosing the current direction *extends* it without a yellow phase |
| **Reward** | −(vehicle-seconds of waiting since the last decision) / 100, minus a penalty for every second any approach exceeds the 90 s red-wait limit |
| **Discount** | γ = 0.997 per simulated second (horizon ≈ 5 min) |

Decisions happen at uneven intervals (10–35 s), so this is a **semi-Markov** decision process.
The target discounts by the real elapsed time τ:

    y = r + γ^τ · max_a' Q_target(s', a')

### Deep Q-Network (`smart_signal/rl_agent.py → DQNAgent`)
- A small neural network (19 + 3 inputs → 64 → 32 → 1, ReLU), written in NumPy — no TensorFlow/PyTorch needed.
- **Direction-symmetric**: the same network scores all four approaches, so experience at North transfers to South, East and West.
- **Experience replay** (30,000 transitions) and a **target network** (synced every 500 updates) for stable learning; Huber loss, Adam optimiser.
- **Guided exploration**: early in training the agent sometimes follows the priority rule (50 % → 0 %) as well as acting randomly (ε 0.30 → 0.05). Q-learning is off-policy, so it learns from those actions too, and the junction doesn't collapse into gridlock during early training.

Training curve (`models/training_log_dqn.csv`): average wait fell from **68 s** (first 100 episodes) to **37 s** (last 100).

### Safety layer — RL only *proposes*
Every proposal passes through `safety_validator.py`, and the simulator enforces the light sequence itself:

1. **Emergency** always wins — through yellow + all-red, then held green until cleared.
2. An approach that has waited **> 90 s** is served next.
3. The same green can be **extended at most twice** while others wait.
4. Green is clipped to **10–30 s**.
5. **Yellow (3 s) and all-red (2 s)** on every change of direction; two greens can never be on together.

The Q-value is updated with the action that was actually executed, so the agent learns the consequences of the safety rules too.

---

## Project structure

```
SmartChauraha/
├── traffic_backend.py        Flask API + live simulation thread
├── traffic_frontend.html     React UI (open directly in a browser)
├── train_rl.py               offline training
├── evaluate.py               fair comparison of all controllers
├── smart_signal/             SmartSignal Engine
│   ├── config.py             every tunable number
│   ├── simulator.py          queue-based intersection (shared by live app and training)
│   ├── rl_agent.py           DQN, linear and tabular Q-learning agents
│   ├── priority_engine.py    weighted multi-factor score (design-doc Stage A)
│   ├── green_optimizer.py    queue-clearance green time (Stage B)
│   ├── safety_validator.py   hard rules (Stage C + safety)
│   ├── decision_engine.py    controller → safety → simulator, plus explanations
│   └── training.py           training / evaluation loops
├── models/                   trained agents (JSON) + learning curves (CSV)
├── results/                  evaluation table, CSV, chart
├── tests/                    unit tests
└── requirements.txt
```

---

## UI guide

- **Signal Logic** — switch between RL agent, Priority rule, Original rule and Fixed time. Live metrics reset on each switch so controllers can be compared.
- **Online learning** — the agent keeps training on live traffic (changes stay in memory; the saved model is untouched).
- **Simulation speed** — 1×, 2×, 5×, 10×.
- **RL Decision panel** — best Q-value per direction, the chosen action, and whether a safety rule changed it.
- **Decision log** — the last 8 decisions with reasons.
- **Queue override / Time of day / Emergency / Manual** — as before, now with safe transitions.

---

## API reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/api/state` | GET | — | Full snapshot (all v2 fields kept, plus queue, waits, decision, Q-values, metrics, evaluation) |
| `/api/emergency` | POST | `{active, direction}` | Safe emergency preemption / clear |
| `/api/mode` | POST | `{mode}` | `AUTO` or `MANUAL` |
| `/api/manual_signal` | POST | `{direction}` | Officer override (via yellow + all-red) |
| `/api/time_of_day` | POST | `{time_of_day}` | `PEAK` / `NORMAL` / `NIGHT` — changes arrival rates |
| `/api/density` | POST | `{direction, value}` | Set a queue (0–100 % of 40 vehicles) |
| `/api/controller` | POST | `{controller}` | `RL` / `PRIORITY` / `LEGACY` / `FIXED` |
| `/api/rl/learning` | POST | `{enabled}` | Online learning on/off |
| `/api/speed` | POST | `{speed}` | 1, 2, 5 or 10 |
| `/api/metrics/reset` | POST | — | Reset live metrics |
| `/api/smart-signal/optimize` | POST | see below | Stateless decision service |

```json
POST /api/smart-signal/optimize
{
  "queue":    {"NORTH": 18, "SOUTH": 9, "EAST": 24, "WEST": 5},
  "avg_wait": {"NORTH": 45, "SOUTH": 30, "EAST": 70, "WEST": 20},
  "red_wait": {"NORTH": 40, "SOUTH": 25, "EAST": 65, "WEST": 15},
  "emergency": null, "current_green": "NORTH",
  "time_of_day": "PEAK", "controller": "RL"
}
→ {"direction": "EAST", "green": 30, "safety_rule": null, "reason": "...",
   "priority_scores": {...}, "rl": {"per_direction": {...}}}
```

---

## Limitations and future work

- One isolated junction; one direction green at a time (no protected turns or combined N–S phases).
- Simulated traffic with assumed arrival rates; a real deployment needs camera/loop-detector counts and calibration.
- Natural next steps: combined phases (N+S together), multi-junction coordination (multi-agent RL), a pedestrian phase, and training on real traffic counts.
