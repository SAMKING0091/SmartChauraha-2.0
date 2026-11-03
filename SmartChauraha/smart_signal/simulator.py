"""
Queue-based 4-way intersection simulator.

The same class is used by
  * the live Flask demo (one tick per real second), and
  * RL training / evaluation (thousands of ticks per real second).

Each vehicle is stored as its arrival time, so real waiting times are known.

Signal cycle enforced here (cannot be skipped by any controller):
    GREEN --> YELLOW (3 s) --> ALL_RED (2 s) --> GREEN (next direction)
A controller may also *extend* the current green, which needs no yellow.
"""

import math
import random
from collections import deque

from . import config as C

DIRS = C.DIRECTIONS


class Intersection:
    def __init__(self, seed=None, time_of_day="NORMAL", initial_queue=None):
        self.rng = random.Random(seed)
        self.time_of_day = time_of_day
        self.t = 0
        self.queues = {d: deque() for d in DIRS}
        self.phase = "ALL_RED"
        self.active = None              # direction that has / last had green
        self.timer = 1
        self.max_timer = 1
        self.pending = None             # (direction, green) waiting for clearance
        self.consecutive = 0            # green extensions in a row
        self.last_green_end = {d: 0 for d in DIRS}
        self.override = None            # {"direction": d, "kind": "EMERGENCY"|"MANUAL"}
        self.cycle_count = 0
        self._discharge_credit = 0.0

        # Metrics
        self.departed = 0
        self.total_wait = 0.0           # sum of waits of departed vehicles
        self.max_wait = 0
        self.arrived = 0
        # Accumulators read (and reset) by the RL agent between decisions
        self.delay_acc = 0.0            # vehicle-seconds of queueing
        self.starve_acc = 0             # seconds with some queue over MAX_RED_WAIT

        if initial_queue:
            for d, n in initial_queue.items():
                self.set_queue(d, n)

    # ── observations ───────────────────────────────────────────
    def queue_len(self, d):
        return len(self.queues[d])

    def avg_wait(self, d):
        q = self.queues[d]
        if not q:
            return 0.0
        return sum(self.t - a for a in q) / len(q)

    def head_wait(self, d):
        q = self.queues[d]
        return self.t - q[0] if q else 0

    def red_wait(self, d):
        """Seconds since direction d last had green (0 while green)."""
        if self.phase == "GREEN" and self.active == d:
            return 0
        if not self.queues[d]:
            return 0
        return self.t - max(self.last_green_end[d], self.queues[d][0])

    def density(self, d):
        return min(100, round(self.queue_len(d) / C.MAX_QUEUE * 100))

    def signals(self):
        s = {d: "RED" for d in DIRS}
        if self.phase == "GREEN":
            s[self.active] = "GREEN"
        elif self.phase == "YELLOW":
            s[self.active] = "YELLOW"
        return s

    def snapshot(self):
        """Plain-dict view consumed by decision engine and controllers."""
        return {
            "t": self.t,
            "phase": self.phase,
            "active": self.active,
            "consecutive": self.consecutive,
            "queue": {d: self.queue_len(d) for d in DIRS},
            "avg_wait": {d: round(self.avg_wait(d), 1) for d in DIRS},
            "red_wait": {d: self.red_wait(d) for d in DIRS},
            "time_of_day": self.time_of_day,
            "emergency": self.override["direction"]
            if self.override and self.override["kind"] == "EMERGENCY" else None,
        }

    def metrics(self):
        return {
            "avg_wait": round(self.total_wait / self.departed, 2) if self.departed else 0.0,
            "max_wait": self.max_wait,
            "throughput": self.departed,
            "arrived": self.arrived,
            "in_queue": sum(self.queue_len(d) for d in DIRS),
            "sim_time": self.t,
        }

    def reset_metrics(self):
        self.departed = self.arrived = self.max_wait = 0
        self.total_wait = 0.0
        self.metrics_since = self.t

    # ── manipulation (used by UI sliders / tests) ──────────────
    def set_queue(self, d, n):
        n = max(0, min(C.QUEUE_CAP, int(n)))
        q = self.queues[d]
        while len(q) > n:
            q.pop()                      # remove newest
        while len(q) < n:
            q.append(self.t)             # add vehicles arriving now
            self.arrived += 1

    # ── internals ──────────────────────────────────────────────
    def _poisson(self, lam):
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1

    def _arrivals(self):
        rates = C.ARRIVAL_RATES[self.time_of_day]
        for d in DIRS:
            for _ in range(self._poisson(rates[d])):
                if len(self.queues[d]) < C.QUEUE_CAP:
                    self.queues[d].append(self.t)
                    self.arrived += 1

    def _discharge(self):
        self._discharge_credit += C.SATURATION_FLOW
        q = self.queues[self.active]
        while self._discharge_credit >= 1 and q:
            arr = q.popleft()
            w = self.t - arr
            self.total_wait += w
            self.max_wait = max(self.max_wait, w)
            self.departed += 1
            self._discharge_credit -= 1
        if not q:
            self._discharge_credit = 0.0

    def _start_green(self, direction, green):
        if direction != self.active:
            self.consecutive = 0
            self.cycle_count += 1
        self.active = direction
        self.phase = "GREEN"
        self.timer = self.max_timer = green
        self.pending = None

    def _held(self):
        """True while an override keeps the current green from ending."""
        return (self.override is not None and self.phase == "GREEN"
                and self.active == self.override["direction"])

    # ── main step ──────────────────────────────────────────────
    def tick(self):
        """
        Advance one second.  Returns "DECIDE" when the controller must
        choose the next (direction, green); otherwise None.
        """
        self.t += 1
        self._arrivals()
        if self.phase == "GREEN":
            self._discharge()

        total_q = 0
        starving = False
        for d in DIRS:
            n = len(self.queues[d])
            total_q += n
            if n and self.red_wait(d) > C.MAX_RED_WAIT:
                starving = True
        self.delay_acc += total_q
        self.starve_acc += 1 if starving else 0

        if self._held():
            return None

        self.timer -= 1
        if self.timer > 0:
            return None

        if self.phase == "GREEN":
            return "DECIDE"
        if self.phase == "YELLOW":
            self.phase = "ALL_RED"
            self.timer = self.max_timer = C.ALL_RED_TIME
            return None
        # ALL_RED finished
        if self.pending:
            self._start_green(*self.pending)
            return None
        return "DECIDE"

    def apply(self, direction, green):
        """
        Carry out a controller decision safely.
          * same direction while green  -> extend, no yellow needed
          * different direction         -> yellow + all-red first
          * from all-red                -> straight to green
        """
        if self.phase == "GREEN" and direction == self.active:
            self.consecutive += 1
            self.timer = self.max_timer = green
        elif self.phase == "GREEN":
            self.last_green_end[self.active] = self.t
            self.pending = (direction, green)
            self.phase = "YELLOW"
            self.timer = self.max_timer = C.YELLOW_TIME
        else:
            self._start_green(direction, green)

    def preempt(self, direction, kind="EMERGENCY"):
        """
        Safe preemption: end the conflicting green through yellow + all-red
        on the next tick instead of switching lights instantly.
        """
        self.override = {"direction": direction, "kind": kind}
        if self.phase == "GREEN" and self.active != direction:
            self.timer = 1
        elif self.phase in ("YELLOW", "ALL_RED") and self.pending:
            self.pending = (direction, self.pending[1])

    def release(self):
        self.override = None
        if self.phase == "GREEN":
            self.timer = min(self.timer, 1)
