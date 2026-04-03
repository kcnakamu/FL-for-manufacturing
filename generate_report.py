"""
generate_report.py
Run after FL training to compile all client results into a summary report.

Usage:
    python generate_report.py                  # uses latest fl_runs/<timestamp>/
    python generate_report.py 20260403_070643  # specify a run
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime


# ── Find run dir ──────────────────────────────────────────────────────────────
fl_runs = Path("fl_runs")
if len(sys.argv) > 1:
    run_root = fl_runs / sys.argv[1]
else:
    # Pick the most recent timestamp folder
    candidates = sorted([d for d in fl_runs.iterdir() if d.is_dir()])
    if not candidates:
        raise FileNotFoundError("No runs found in fl_runs/")
    run_root = candidates[-1]

print(f"Generating report for: {run_root}")

# ── Collect all results.csv files ─────────────────────────────────────────────
records = []
for csv_path in sorted(run_root.glob("round_*/client_*/results.csv")):
    parts  = csv_path.parts
    # Extract round and client from path
    round_dir  = [p for p in parts if p.startswith("round_")][0]
    client_dir = [p for p in parts if p.startswith("client_")][0]
    round_num  = int(round_dir.split("_")[1])
    client_id  = client_dir.split("_")[1]

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1]  # last epoch of this round

    records.append({
        "round":       round_num,
        "client":      f"client_{client_id}",
        "mAP50":       float(last.get("metrics/mAP50(B)",   0)),
        "mAP50-95":    float(last.get("metrics/mAP50-95(B)", 0)),
        "precision":   float(last.get("metrics/precision(B)", 0)),
        "recall":      float(last.get("metrics/recall(B)",    0)),
        "box_loss":    float(last.get("train/box_loss",        0)),
        "cls_loss":    float(last.get("train/cls_loss",        0)),
    })

summary = pd.DataFrame(records).sort_values(["round", "client"])
print(summary.to_string(index=False))

# ── Aggregated per-round stats ────────────────────────────────────────────────
agg = summary.groupby("round").agg(
    mAP50_mean    =("mAP50",    "mean"),
    mAP50_std     =("mAP50",    "std"),
    mAP5095_mean  =("mAP50-95", "mean"),
    mAP5095_std   =("mAP50-95", "std"),
    box_loss_mean =("box_loss", "mean"),
    cls_loss_mean =("cls_loss", "mean"),
).reset_index()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"FL Training Report — {run_root.name}", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

rounds = agg["round"]
clients = summary["client"].unique()
colors  = plt.cm.tab10.colors

# 1. mAP50 per round (aggregated mean ± std)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(rounds, agg["mAP50_mean"], marker="o", color="steelblue", label="Mean")
ax1.fill_between(rounds,
    agg["mAP50_mean"] - agg["mAP50_std"].fillna(0),
    agg["mAP50_mean"] + agg["mAP50_std"].fillna(0),
    alpha=0.2, color="steelblue")
ax1.set_title("mAP50 (aggregated)")
ax1.set_xlabel("Round"); ax1.set_ylabel("mAP50")
ax1.set_xticks(rounds); ax1.legend()

# 2. mAP50 per client per round
ax2 = fig.add_subplot(gs[0, 1])
for i, client in enumerate(clients):
    cdf = summary[summary["client"] == client]
    ax2.plot(cdf["round"], cdf["mAP50"], marker="o",
             color=colors[i], label=client)
ax2.set_title("mAP50 per client")
ax2.set_xlabel("Round"); ax2.set_ylabel("mAP50")
ax2.set_xticks(rounds); ax2.legend()

# 3. Box loss per client
ax3 = fig.add_subplot(gs[1, 0])
for i, client in enumerate(clients):
    cdf = summary[summary["client"] == client]
    ax3.plot(cdf["round"], cdf["box_loss"], marker="o",
             color=colors[i], label=client)
ax3.set_title("Box loss per client")
ax3.set_xlabel("Round"); ax3.set_ylabel("Box loss")
ax3.set_xticks(rounds); ax3.legend()

# 4. Precision & Recall (aggregated)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(rounds, summary.groupby("round")["precision"].mean().reindex(rounds).values,
         marker="o", color="green", label="Precision")
ax4.plot(rounds,
         summary.groupby("round")["recall"].mean().reindex(rounds).values,
         marker="s", color="orange", label="Recall")
ax4.set_title("Precision & Recall (aggregated)")
ax4.set_xlabel("Round"); ax4.set_ylabel("Score")
ax4.set_xticks(rounds); ax4.legend()

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = run_root
plot_path   = out_dir / "fl_report.png"
csv_path_out = out_dir / "fl_summary.csv"

fig.savefig(plot_path, dpi=150, bbox_inches="tight")
summary.to_csv(csv_path_out, index=False)

print(f"\nSaved:\n  {plot_path}\n  {csv_path_out}")
plt.close()