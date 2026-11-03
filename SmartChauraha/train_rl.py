"""
Train the SmartSignal RL agent offline on the simulator.

    python train_rl.py                        # DQN, 3000 episodes (~12 min)
    python train_rl.py --model linear         # linear Q-learning (~2 min)
    python train_rl.py --model tabular        # tabular Q-learning (~1 min)
    python train_rl.py --episodes 500         # quick run for testing

Each episode = one simulated hour at a randomly chosen time of day
(50 % PEAK, 30 % NORMAL, 20 % NIGHT) with random starting queues.

Outputs
    models/rl_<model>.json             trained agent (loaded by the backend)
    models/training_log_<model>.csv    learning curve for the report
"""

import argparse
import csv
import os
import time

from smart_signal import config as C
from smart_signal.rl_agent import DEFAULT_EPISODES, make_agent, model_path
from smart_signal.training import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["dqn", "linear", "tabular"], default=C.RL_MODEL)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seconds", type=int, default=3600, help="simulated seconds per episode")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    episodes = args.episodes or DEFAULT_EPISODES[args.model]

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    print(f"Training {args.model} agent: {episodes} episodes x {args.seconds}s simulated")
    print(f"{'episode':>8} {'epsilon':>8} {'avg wait (s)':>13} {'elapsed':>9}")

    def progress(row):
        print(f"{row['episode']:>8} {row['epsilon']:>8.3f} {row['avg_wait']:>13.2f} {row['elapsed_s']:>8.0f}s",
              flush=True)

    t0 = time.time()
    agent, history = train(episodes=episodes, seconds=args.seconds, seed=args.seed,
                           agent=make_agent(args.model, seed=args.seed),
                           log_every=max(1, episodes // 30), progress=progress)
    agent.meta = {"trained_on": time.strftime("%Y-%m-%d %H:%M"), "episodes": episodes,
                  "seconds_per_episode": args.seconds, "train_time_s": round(time.time() - t0)}
    path = model_path(args.model)
    agent.save(path)

    log = os.path.join(C.MODEL_DIR, f"training_log_{args.model}.csv")
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "epsilon", "avg_wait", "states", "elapsed_s"])
        w.writeheader()
        w.writerows(history)
    print(f"\nSaved agent -> {path}\nSaved learning curve -> {log}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
