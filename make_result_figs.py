"""Regenerate the leaderboard-dependent result figures from one results table.

Reads model_results.csv (written by make_model_figs.py: Model, Accuracy,
F1-Score, ROC-AUC, Seconds) and rebuilds, in the deck's palette:

  * figures/fig_leaderboard_auc.png  -- ranked ROC-AUC bar chart
  * figures/fig_f1_vs_auc.png        -- F1 vs. ROC-AUC agreement scatter
  * figures/fig_acc_vs_time.png      -- accuracy vs. training cost (log time)

fig_stages.png and fig_classbalance.png are independent of the leaderboard and
are left untouched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RED, DARK, GREY = "#9B1B30", "#1f1f1f", "#9aa0a6"

plt.rcParams.update({
    "font.size": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

res = pd.read_csv("model_results.csv").sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
SHORT = {"Logistic Regression": "Logistic Reg."}
res["disp"] = res["Model"].map(lambda m: SHORT.get(m, m))

# ---------------------------------------------------------- 1. leaderboard bars
fig, ax = plt.subplots(figsize=(9, 5.2))
order = res.iloc[::-1]                      # best on top
colors = [RED if m == res["Model"].iloc[0] else GREY for m in order["Model"]]
bars = ax.barh(order["disp"], order["ROC-AUC"], color=colors)
ax.set_xlim(0.93, res["ROC-AUC"].max() + 0.004)
for b, v in zip(bars, order["ROC-AUC"]):
    ax.text(v - 0.0008, b.get_y() + b.get_height() / 2, f"{v:.4f}",
            va="center", ha="right", color="white", fontweight="bold", fontsize=13)
ax.set_xlabel("ROC-AUC  (5-fold Stratified CV)")
ax.set_title("Model leaderboard — ROC-AUC", fontweight="bold", fontsize=18)
fig.tight_layout()
fig.savefig("figures/fig_leaderboard_auc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- 2. F1 vs ROC-AUC
fig, ax = plt.subplots(figsize=(9, 5.4))
top = res["Model"].iloc[0]
for _, r in res.iterrows():
    c = RED if r["Model"] == top else DARK
    ax.scatter(r["ROC-AUC"], r["F1-Score"], s=90, color=c, zorder=5)
    ax.annotate(r["disp"], (r["ROC-AUC"], r["F1-Score"]),
                textcoords="offset points", xytext=(7, 4),
                fontsize=11, color=c, fontweight="bold")
ax.set_xlabel("ROC-AUC"); ax.set_ylabel("F1-Score")
ax.set_title("F1 vs. ROC-AUC agreement", fontweight="bold", fontsize=18)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/fig_f1_vs_auc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("regenerated fig_leaderboard_auc.png, fig_f1_vs_auc.png")
print(res[["Model", "Accuracy", "F1-Score", "ROC-AUC", "Seconds"]].round(4).to_string(index=False))
