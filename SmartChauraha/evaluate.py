"""
Compare every controller on identical traffic.

    python evaluate.py                 # 20 seeds x 3 times of day x 1 h each
    python evaluate.py --seeds 50      # tighter averages

The same random seed gives every controller exactly the same vehicle
arrivals, so differences come from the signal logic alone.

Outputs
    results/evaluation.csv     one row per controller x time of day
    results/evaluation.md      table ready to paste into the report
    results/evaluation.png     bar chart (only if matplotlib is installed)
"""

import argparse
import csv
import os

from smart_signal.rl_agent import load_agent, MODELS
from smart_signal.training import evaluate

TODS = ("PEAK", "NORMAL", "NIGHT")
LABELS = {
    "FIXED": "Fixed-time (20 s round robin)",
    "LEGACY": "Original SmartChauraha rule",
    "PRIORITY": "Weighted priority + clearance",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seconds", type=int, default=3600)
    args = ap.parse_args()
    seeds = range(5000, 5000 + args.seeds)   # held out: never used while tuning

    rows = []
    base = evaluate(load_agent("tabular")[0], seeds, args.seconds, ("FIXED", "LEGACY", "PRIORITY"))
    for ctrl, per_tod in base.items():
        rows += [{"controller": LABELS[ctrl], "key": ctrl, "time_of_day": t, **per_tod[t]} for t in TODS]

    for model in MODELS:
        agent, ok = load_agent(model)
        if not ok:
            print(f"(skipping RL-{model}: no trained model - run  python train_rl.py --model {model})")
            continue
        res = evaluate(agent, seeds, args.seconds, ("RL",))["RL"]
        name = {"dqn": "RL - Deep Q-Network", "linear": "RL - linear Q-learning",
                "tabular": "RL - tabular Q-learning"}[model]
        rows += [{"controller": name, "key": f"RL_{model}", "time_of_day": t, **res[t]} for t in TODS]

    os.makedirs("results", exist_ok=True)
    fields = ["controller", "key", "time_of_day", "avg_wait", "max_wait", "throughput", "left_in_queue"]
    with open("results/evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Markdown table: average wait per vehicle (s), lower is better
    ctrls = list(dict.fromkeys(r["controller"] for r in rows))
    get = {(r["controller"], r["time_of_day"]): r for r in rows}
    lines = [f"Average wait per vehicle in seconds (lower is better). "
             f"{args.seeds} seeds x {args.seconds // 60} min per time of day.\n",
             "| Controller | Peak | Normal | Night | Worst single wait (peak) |",
             "|---|---|---|---|---|"]
    for c in ctrls:
        cells = []
        for t in TODS:
            v = get[(c, t)]["avg_wait"]
            best = min(get[(x, t)]["avg_wait"] for x in ctrls)
            cells.append(f"**{v:.1f}**" if v == best else f"{v:.1f}")
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {get[(c, 'PEAK')]['max_wait']} |")
    md = "\n".join(lines)
    with open("results/evaluation.md", "w") as f:
        f.write(md + "\n")
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / len(ctrls)
        for i, c in enumerate(ctrls):
            ax.bar([x + i * width for x in range(3)], [get[(c, t)]["avg_wait"] for t in TODS],
                   width, label=c)
        ax.set_xticks([x + width * (len(ctrls) - 1) / 2 for x in range(3)])
        ax.set_xticklabels(["Peak", "Normal", "Night"])
        ax.set_ylabel("Average wait per vehicle (s)")
        ax.set_title("SmartChauraha controllers on identical traffic")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("results/evaluation.png", dpi=150)
        print("\nChart -> results/evaluation.png")
    except ImportError:
        print("\n(matplotlib not installed - skipped chart)")
    print("CSV   -> results/evaluation.csv\nTable -> results/evaluation.md")


if __name__ == "__main__":
    main()
