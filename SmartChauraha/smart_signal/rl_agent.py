"""
Reinforcement-learning signal controller: tabular Q-learning (semi-MDP).

State  s = (queue bucket N, S, E, W,  current green direction,  starving direction)
           queue buckets: 0-2 | 3-8 | 9-18 | 19+         -> 4^4 x 5 x 5 states
Action a = (direction to serve, green duration in {10, 20, 30} s) -> 12 actions
Reward r = -(vehicle-seconds of queueing since last decision) / 100
           - 2 x (seconds in which some queue exceeded MAX_RED_WAIT)

Decisions happen at irregular intervals (10-35 s), so the update discounts
by the actual elapsed time tau (semi-Markov Q-learning):

    Q(s,a) <- Q(s,a) + alpha [ r + gamma^tau * max_a' Q(s',a') - Q(s,a) ]

The agent only *proposes*. Every proposal passes through safety_validator,
and Q is updated with the action that was actually executed.
"""

import json
import os
import random

from . import config as C
from .green_optimizer import queue_clearance_green
from .priority_engine import priority_scores

DIRS = C.DIRECTIONS
N_ACTIONS = len(DIRS) * len(C.GREEN_CHOICES)


def action_to_pair(a):
    return DIRS[a // len(C.GREEN_CHOICES)], C.GREEN_CHOICES[a % len(C.GREEN_CHOICES)]


def pair_to_action(direction, green):
    gi = min(range(len(C.GREEN_CHOICES)), key=lambda i: abs(C.GREEN_CHOICES[i] - green))
    return DIRS.index(direction) * len(C.GREEN_CHOICES) + gi


def bucket(q):
    for i, edge in enumerate(C.QUEUE_BUCKETS):
        if q < edge:
            return i
    return len(C.QUEUE_BUCKETS)


def encode_state(snap):
    qb = tuple(bucket(snap["queue"][d]) for d in DIRS)
    active = DIRS.index(snap["active"]) if snap["active"] in DIRS and snap["phase"] == "GREEN" else 4
    starving, worst = 4, C.STARVE_WARN
    for i, d in enumerate(DIRS):
        if snap["queue"][d] and snap["red_wait"][d] > worst:
            starving, worst = i, snap["red_wait"][d]
    return qb + (active, starving)


def state_key(s):
    return ",".join(map(str, s))


class QLearningAgent:
    kind = "tabular"

    def __init__(self, alpha=C.ALPHA, gamma=C.GAMMA_PER_SECOND, epsilon=C.EPSILON_LIVE, seed=None):
        self.q = {}                     # state_key -> [Q for each action]
        self.n = {}                     # state_key -> [visit count for each action]
        self.guide = 0.0                # prob. of following the rule-based guide (training only)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.episodes = 0
        self.updates = 0
        self.last_reward = 0.0
        self.meta = {}
        self._prev = None               # (state_key, action, t)

    # ── table access ───────────────────────────────────────────
    def values(self, s):
        k = state_key(s)
        if k not in self.q:
            self.q[k] = [0.0] * N_ACTIONS
        return self.q[k]

    def known(self, s):
        return state_key(s) in self.q

    # ── acting ─────────────────────────────────────────────────
    def counts(self, s):
        return self.n.get(state_key(s), [0] * N_ACTIONS)

    def select(self, snap, allowed_dirs, epsilon=None):
        """
        Returns an action index, or None meaning "no experience here -
        use the rule-based fallback" (also used as the training guide).
        """
        eps = self.epsilon if epsilon is None else epsilon
        legal = [a for a in range(N_ACTIONS) if action_to_pair(a)[0] in allowed_dirs]
        r = self.rng.random()
        if r < eps:
            return self.rng.choice(legal)
        if r < eps + self.guide:
            return None
        s = encode_state(snap)
        qs, ns = self.q.get(state_key(s)), self.counts(s)
        tried = [a for a in legal if ns[a] > 0]
        if not qs or not tried:
            return None
        best = max(qs[a] for a in tried)
        return self.rng.choice([a for a in tried if qs[a] == best])

    def state_value(self, s):
        """max_a Q(s,a) over actions actually tried (untried values are meaningless)."""
        k = state_key(s)
        ns = self.n.get(k)
        if not ns or not any(ns):
            return None
        return max(v for v, c in zip(self.q[k], ns) if c > 0)

    # ── learning ───────────────────────────────────────────────
    def observe(self, sim, learn=True):
        """
        Called at every decision point *before* choosing the next action.
        Closes the previous transition: reward since last decision -> update.
        """
        reward = -(sim.delay_acc / C.REWARD_SCALE) - C.STARVATION_PENALTY * sim.starve_acc
        sim.delay_acc, sim.starve_acc = 0.0, 0
        if self._prev is None:
            return None
        k_prev, a_prev, t_prev = self._prev
        self._prev = None
        self.last_reward = round(reward, 3)
        if not learn:
            return reward
        tau = max(1, sim.t - t_prev)
        v_next = self.state_value(encode_state(sim.snapshot()))
        row = self.q.setdefault(k_prev, [0.0] * N_ACTIONS)
        cnt = self.n.setdefault(k_prev, [0] * N_ACTIONS)
        if v_next is None:
            # Unseen next state: assume this step's cost keeps recurring.
            v_next = reward / (1.0 - self.gamma ** tau)
        target = reward + (self.gamma ** tau) * v_next
        if cnt[a_prev] == 0:
            row[a_prev] = target                 # first visit: take the sample directly
        else:
            row[a_prev] += self.alpha * (target - row[a_prev])
        cnt[a_prev] += 1
        self.updates += 1
        return reward

    def remember(self, snap, executed_direction, executed_green, t):
        """Store the action that was really executed (after safety checks)."""
        self._prev = (state_key(encode_state(snap)),
                      pair_to_action(executed_direction, executed_green), t)

    def reset_episode(self):
        self._prev = None

    # ── explanation for the UI ─────────────────────────────────
    def explain(self, snap):
        s = encode_state(snap)
        qs = self.q.get(state_key(s), [0.0] * N_ACTIONS)
        per_dir = {}
        ns_ = self.counts(s)
        for a, v in enumerate(qs):
            if not ns_[a]:
                continue
            d, g = action_to_pair(a)
            if d not in per_dir or v > per_dir[d]["q"]:
                per_dir[d] = {"q": round(v, 3), "green": g}
        ns = self.counts(s)
        return {"state": list(s), "seen_in_training": any(ns), "per_direction": per_dir}

    # ── persistence ────────────────────────────────────────────
    def save(self, path=C.Q_TABLE_PATH):
        data = {"episodes": self.episodes, "updates": self.updates,
                "green_choices": C.GREEN_CHOICES, "meta": self.meta, "q": self.q, "n": self.n}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path=C.Q_TABLE_PATH):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        if data.get("green_choices") != C.GREEN_CHOICES:
            return False                   # config changed -> table is stale
        self.q = data["q"]
        self.n = data.get("n", {k: [1] * N_ACTIONS for k in self.q})
        self.episodes = data.get("episodes", 0)
        self.updates = data.get("updates", 0)
        self.meta = data.get("meta", {})
        return True


# ════════════════════════════════════════════════════════════════
# Q-learning with linear function approximation
# ════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "bias", "own_queue", "own_queue_sq", "own_avg_wait", "own_red_wait",
    "is_current_green", "current_green_x_queue", "own_empty",
    "others_total_queue", "others_max_queue", "others_max_red_wait",
    "total_queue", "total_queue_sq",
    "lead_over_longest_other", "is_longest_queue", "is_longest_red_wait",
    "priority_score", "is_top_priority", "clearance_green",
]


def features(snap, d):
    """
    Direction-symmetric features for serving direction d.
    The same weights apply to all four approaches, so what the agent learns
    about North transfers to South, East and West.
    """
    q = snap["queue"]
    others = [x for x in DIRS if x != d]
    qd = q[d] / 40.0
    pr = snap.get("_priority") or priority_scores(snap)
    cur = 1.0 if (snap["phase"] == "GREEN" and snap["active"] == d) else 0.0
    tot = sum(q.values()) / 120.0
    return [
        1.0, qd, qd * qd,
        snap["avg_wait"][d] / 90.0,
        snap["red_wait"][d] / 90.0,
        cur, cur * qd,
        1.0 if q[d] == 0 else 0.0,
        sum(q[x] for x in others) / 120.0,
        max(q[x] for x in others) / 40.0,
        max(snap["red_wait"][x] for x in others) / 90.0,
        tot, tot * tot,
        (q[d] - max(q[x] for x in others)) / 40.0,
        1.0 if q[d] >= max(q[x] for x in others) and q[d] > 0 else 0.0,
        1.0 if snap["red_wait"][d] >= max(snap["red_wait"][x] for x in others)
        and snap["red_wait"][d] > 0 else 0.0,
        # Domain knowledge from the rule-based engine, offered as *inputs*;
        # the agent learns how much weight they deserve.
        pr[d]["score"],
        1.0 if pr[d]["score"] >= max(pr[x]["score"] for x in others) else 0.0,
        queue_clearance_green(q[d], d, snap.get("time_of_day", "NORMAL")) / C.MAX_GREEN,
    ]


class LinearQAgent:
    """
    Q(s, (d, g)) = w_g . phi(s, d)       one weight vector per green choice g

    Semi-gradient SMDP Q-learning:
        w_g <- w_g + alpha [ r + gamma^tau max_a' Q(s',a') - Q(s,a) ] phi(s,d)
    """
    kind = "linear"

    def __init__(self, alpha=0.01, gamma=C.GAMMA_PER_SECOND, epsilon=C.EPSILON_LIVE, seed=None):
        nf = len(FEATURE_NAMES)
        self.w = {str(g): [0.0] * nf for g in C.GREEN_CHOICES}
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.guide = 0.0
        self.rng = random.Random(seed)
        self.episodes = self.updates = 0
        self.last_reward = 0.0
        self.meta = {}
        self._prev = None               # (phi, g, t)
        self.trained = False

    def qvalue(self, phi, g):
        w = self.w[str(g)]
        return sum(a * b for a, b in zip(w, phi))

    def _all_q(self, snap, allowed_dirs):
        snap = dict(snap, _priority=priority_scores(snap))
        out = []
        for d in allowed_dirs:
            phi = features(snap, d)
            for g in C.GREEN_CHOICES:
                out.append((self.qvalue(phi, g), d, g))
        return out

    def select(self, snap, allowed_dirs, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        legal = [a for a in range(N_ACTIONS) if action_to_pair(a)[0] in allowed_dirs]
        r = self.rng.random()
        if r < eps:
            return self.rng.choice(legal)
        if r < eps + self.guide or not self.trained:
            return None
        _, d, g = max(self._all_q(snap, allowed_dirs))
        return pair_to_action(d, g)

    def observe(self, sim, learn=True):
        reward = -(sim.delay_acc / C.REWARD_SCALE) - C.STARVATION_PENALTY * sim.starve_acc
        sim.delay_acc, sim.starve_acc = 0.0, 0
        if self._prev is None:
            return None
        phi, g, t_prev = self._prev
        self._prev = None
        self.last_reward = round(reward, 3)
        if not learn:
            return reward
        tau = max(1, sim.t - t_prev)
        snap = sim.snapshot()
        from .safety_validator import allowed_directions
        v_next = max(q for q, _, _ in self._all_q(snap, allowed_directions(snap)))
        err = reward + (self.gamma ** tau) * v_next - self.qvalue(phi, g)
        err = max(-50.0, min(50.0, err))          # clip TD error for stability
        w = self.w[str(g)]
        for i, x in enumerate(phi):
            w[i] += self.alpha * err * x
        self.updates += 1
        self.trained = True
        return reward

    def remember(self, snap, executed_direction, executed_green, t):
        g = min(C.GREEN_CHOICES, key=lambda x: abs(x - executed_green))
        self._prev = (features(snap, executed_direction), g, t)

    def reset_episode(self):
        self._prev = None

    def explain(self, snap):
        per_dir = {}
        for q, d, g in self._all_q(snap, DIRS):
            if d not in per_dir or q > per_dir[d]["q"]:
                per_dir[d] = {"q": round(q, 3), "green": g}
        return {"model": "linear", "seen_in_training": self.trained, "per_direction": per_dir}

    def save(self, path=C.Q_TABLE_PATH):
        with open(path, "w") as f:
            json.dump({"model": "linear", "episodes": self.episodes, "updates": self.updates,
                       "green_choices": C.GREEN_CHOICES, "features": FEATURE_NAMES,
                       "meta": self.meta, "w": self.w}, f, indent=1)

    def load(self, path=C.Q_TABLE_PATH):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        if (data.get("model") != "linear" or data.get("green_choices") != C.GREEN_CHOICES
                or data.get("features") != FEATURE_NAMES):
            return False
        self.w = data["w"]
        self.episodes, self.updates = data.get("episodes", 0), data.get("updates", 0)
        self.meta = data.get("meta", {})
        self.trained = True
        return True


# ════════════════════════════════════════════════════════════════
# Deep Q-Network (NumPy MLP, experience replay, target network)
# ════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Q(s, (d, g)) = MLP( phi(s, d) ++ onehot(g) )      shared across directions

    * Experience replay: transitions are stored and learned from in random
      mini-batches, which breaks the correlation between consecutive steps.
    * Target network: a frozen copy of the network computes the TD target
      and is refreshed every TARGET_SYNC updates (stabilises learning).
    * Semi-MDP target:  y = r + gamma^tau * max_a' Q_target(s', a')
    * Huber loss, Adam optimiser.
    """
    kind = "dqn"
    VALUE_SCALE = 10.0          # rewards divided by this inside the network
    TARGET_SYNC = 500

    def __init__(self, hidden=(64, 32), lr=1e-3, gamma=0.997, epsilon=C.EPSILON_LIVE,
                 batch=64, buffer=30000, seed=None):
        import numpy as np
        self.np = np
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self.n_in = len(FEATURE_NAMES) + len(C.GREEN_CHOICES)
        sizes = [self.n_in, *hidden, 1]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params.append([self.nrng.normal(0, (2.0 / a) ** 0.5, (a, b)).astype(np.float32),
                                np.zeros(b, np.float32)])
        self.target = [[w.copy(), b.copy()] for w, b in self.params]
        self.m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in self.params]
        self.v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in self.params]
        self.lr, self.gamma, self.epsilon, self.batch = lr, gamma, epsilon, batch
        self.guide = 0.0
        self.cap = buffer
        self.buf = []
        self.pos = 0
        self.episodes = self.updates = self.steps = 0
        self.last_reward = 0.0
        self.meta = {}
        self._prev = None
        self.trained = False
        self.warmup = 1000          # transitions collected before learning starts

    # ── network ────────────────────────────────────────────────
    def _inputs(self, snap, pairs):
        np = self.np
        sn = dict(snap, _priority=priority_scores(snap))
        cache = {}
        X = np.zeros((len(pairs), self.n_in), np.float32)
        for i, (d, g) in enumerate(pairs):
            if d not in cache:
                cache[d] = features(sn, d)
            X[i, :len(FEATURE_NAMES)] = cache[d]
            X[i, len(FEATURE_NAMES) + C.GREEN_CHOICES.index(g)] = 1.0
        return X

    def _forward(self, X, params, keep=False):
        np = self.np
        acts = [X]
        h = X
        for i, (w, b) in enumerate(params):
            h = h @ w + b
            if i < len(params) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return (h[:, 0], acts) if keep else h[:, 0]

    def _pairs(self, allowed_dirs):
        return [(d, g) for d in allowed_dirs for g in C.GREEN_CHOICES]

    # ── acting ─────────────────────────────────────────────────
    def select(self, snap, allowed_dirs, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        legal = [a for a in range(N_ACTIONS) if action_to_pair(a)[0] in allowed_dirs]
        r = self.rng.random()
        if r < eps:
            return self.rng.choice(legal)
        if r < eps + self.guide or not self.trained:
            return None
        pairs = self._pairs(allowed_dirs)
        qs = self._forward(self._inputs(snap, pairs), self.params)
        d, g = pairs[int(qs.argmax())]
        return pair_to_action(d, g)

    # ── learning ───────────────────────────────────────────────
    def observe(self, sim, learn=True):
        reward = -(sim.delay_acc / C.REWARD_SCALE) - C.STARVATION_PENALTY * sim.starve_acc
        sim.delay_acc, sim.starve_acc = 0.0, 0
        if self._prev is None:
            return None
        x, t_prev = self._prev
        self._prev = None
        self.last_reward = round(reward, 3)
        if not learn:
            return reward
        from .safety_validator import allowed_directions
        snap = sim.snapshot()
        nx = self._inputs(snap, self._pairs(allowed_directions(snap)))
        tr = (x, reward / self.VALUE_SCALE, self.gamma ** max(1, sim.t - t_prev), nx)
        if len(self.buf) < self.cap:
            self.buf.append(tr)
        else:
            self.buf[self.pos] = tr
        self.pos = (self.pos + 1) % self.cap
        self.steps += 1
        if len(self.buf) >= self.warmup:
            self._train_batch()
        return reward

    def _train_batch(self):
        np = self.np
        idx = self.nrng.integers(0, len(self.buf), self.batch)
        batch = [self.buf[i] for i in idx]
        X = np.stack([b[0] for b in batch])
        R = np.array([b[1] for b in batch], np.float32)
        G = np.array([b[2] for b in batch], np.float32)
        sizes = [len(b[3]) for b in batch]
        NX = np.concatenate([b[3] for b in batch])
        nq = self._forward(NX, self.target)
        bounds = np.cumsum([0] + sizes)
        vmax = np.array([nq[bounds[i]:bounds[i + 1]].max() for i in range(len(batch))], np.float32)
        y = R + G * vmax

        q, acts = self._forward(X, self.params, keep=True)
        err = q - y
        grad = np.clip(err, -1.0, 1.0)[:, None] / len(batch)      # Huber gradient
        self.updates += 1
        t = self.updates
        for i in reversed(range(len(self.params))):
            w, b = self.params[i]
            gw = acts[i].T @ grad
            gb = grad.sum(0)
            if i > 0:
                grad = (grad @ w.T) * (acts[i] > 0)
            for j, gp in enumerate((gw, gb)):
                self.m[i][j] = 0.9 * self.m[i][j] + 0.1 * gp
                self.v[i][j] = 0.999 * self.v[i][j] + 0.001 * gp * gp
                mh = self.m[i][j] / (1 - 0.9 ** t)
                vh = self.v[i][j] / (1 - 0.999 ** t)
                self.params[i][j] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        if t % self.TARGET_SYNC == 0:
            self.target = [[w.copy(), b.copy()] for w, b in self.params]
        self.trained = True

    def remember(self, snap, executed_direction, executed_green, t):
        g = min(C.GREEN_CHOICES, key=lambda x: abs(x - executed_green))
        self._prev = (self._inputs(snap, [(executed_direction, g)])[0], t)

    def reset_episode(self):
        self._prev = None

    # ── explanation & persistence ──────────────────────────────
    def explain(self, snap):
        pairs = self._pairs(DIRS)
        qs = self._forward(self._inputs(snap, pairs), self.params) * self.VALUE_SCALE
        per_dir = {}
        for (d, g), v in zip(pairs, qs.tolist()):
            if d not in per_dir or v > per_dir[d]["q"]:
                per_dir[d] = {"q": round(v, 2), "green": g}
        return {"model": "dqn", "seen_in_training": self.trained, "per_direction": per_dir}

    def save(self, path=C.Q_TABLE_PATH):
        with open(path, "w") as f:
            json.dump({"model": "dqn", "episodes": self.episodes, "updates": self.updates,
                       "green_choices": C.GREEN_CHOICES, "features": FEATURE_NAMES,
                       "meta": self.meta,
                       "params": [[w.tolist(), b.tolist()] for w, b in self.params]}, f)

    def load(self, path=C.Q_TABLE_PATH):
        np = self.np
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        if (data.get("model") != "dqn" or data.get("green_choices") != C.GREEN_CHOICES
                or data.get("features") != FEATURE_NAMES):
            return False
        self.params = [[np.array(w, np.float32), np.array(b, np.float32)] for w, b in data["params"]]
        self.target = [[w.copy(), b.copy()] for w, b in self.params]
        self.episodes, self.updates = data.get("episodes", 0), data.get("updates", 0)
        self.meta = data.get("meta", {})
        self.trained = True
        self.warmup = 16            # already trained: online learning starts almost at once
        return True


# ════════════════════════════════════════════════════════════════
MODELS = {"tabular": QLearningAgent, "linear": LinearQAgent, "dqn": DQNAgent}
DEFAULT_EPISODES = {"tabular": 6000, "linear": 3000, "dqn": 3000}


def model_path(model):
    return os.path.join(C.MODEL_DIR, f"rl_{model}.json")


def make_agent(model=None, seed=None):
    model = model or C.RL_MODEL
    if model == "tabular":
        return QLearningAgent(seed=seed)
    if model == "linear":
        return LinearQAgent(alpha=0.01, seed=seed)
    return DQNAgent(lr=3e-4, seed=seed)


def load_agent(model=None, base_dir="."):
    """Load a trained agent; returns (agent, loaded_ok)."""
    model = model or C.RL_MODEL
    agent = make_agent(model)
    ok = agent.load(os.path.join(base_dir, model_path(model)))
    return agent, ok
