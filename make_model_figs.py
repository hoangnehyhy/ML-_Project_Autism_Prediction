"""Tree-model hyperparameter search + training-diagnostic figures.

Local companion to notebooks/08_hyperparameter_tuning.ipynb. It runs the same
compact grids for the six tree/linear baselines on the unified cleaned train set
(5-fold Stratified CV), keeps the best config per model, and renders:

  * figures/fig_loss_gbdt.png   -- train vs. validation logloss for the GBDT trio
  * figures/fig_tuning_panel.png -- CV AUC vs. a key hyperparameter per model

It also writes a combined results table (tree best rows + the fixed
TabM/TabPFN/Stacking numbers from notebooks 06-07) used by make_result_figs.py
to keep every leaderboard figure consistent.

The heavy deep/foundation models (TabM, TabPFN) are tuned in the notebook on
GPU; their numbers are merged here as constants so the deck still builds locally.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 42
RED, DARK, GREY = "#9B1B30", "#1f1f1f", "#c8c8c8"
BLUE = "#2c5f8a"

plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ----------------------------------------------------------------- data
DATA = Path("data/processed")
train = pd.read_csv(DATA / "train_cleaned.csv")
TARGET = "Class/ASD"
X = train.drop(columns=[TARGET]).values.astype(np.float32)
y = train[TARGET].values.astype(np.int64)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
spw = (y == 0).sum() / (y == 1).sum()
print(f"train: {X.shape} | positives {y.mean():.3f} | scale_pos_weight {spw:.2f}")


def cv_score(make):
    """Mean Acc / F1 / ROC-AUC over the shared 5 folds for one estimator factory."""
    r = cross_validate(make(), X, y, cv=cv,
                       scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
    return (r["test_accuracy"].mean(), r["test_f1"].mean(), r["test_roc_auc"].mean())


def search(name, configs, make, key):
    """Evaluate a compact grid, print a per-config table, return best row + sweep.

    key -- the hyperparameter plotted on the tuning panel (marginal best per value).
    """
    print(f"\n=== {name} ({len(configs)} configs) ===")
    rows = []
    for cfg in configs:
        t0 = time.time()
        acc, f1, auc = cv_score(lambda c=cfg: make(**c))
        rows.append({**cfg, "Accuracy": acc, "F1": f1, "AUC": auc})
        print(f"  {cfg} -> AUC {auc:.4f} F1 {f1:.4f}  ({time.time()-t0:.1f}s)")
    best = max(rows, key=lambda r: r["AUC"])
    # marginal best AUC per value of `key` for the panel
    sweep = {}
    for r in rows:
        sweep[r[key]] = max(sweep.get(r[key], 0), r["AUC"])
    sweep = sorted(sweep.items(), key=lambda kv: (kv[0] is None, kv[0]))
    # time one fit on all data for the cost plot
    bcfg = {k: v for k, v in best.items() if k not in ("Accuracy", "F1", "AUC")}
    t0 = time.time(); make(**bcfg).fit(X, y); fit_t = time.time() - t0
    print(f"  best: {bcfg} -> AUC {best['AUC']:.4f}  (fit {fit_t:.1f}s)")
    return best, sweep, fit_t, bcfg


# ----------------------------------------------------------------- grids
results, sweeps, times = [], {}, {}

best, sweeps["Logistic Regression"], t, _ = search(
    "Logistic Regression",
    [{"C": c} for c in (0.01, 0.05, 0.1, 0.5, 1.0, 3.0)],
    lambda C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    key="C")
results.append(("Logistic Regression", best, t))

best, sweeps["Random Forest"], t, _ = search(
    "Random Forest",
    [{"max_depth": d} for d in (8, 12, 16, 20, None)],
    lambda max_depth: RandomForestClassifier(
        n_estimators=300, max_depth=max_depth, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1),
    key="max_depth")
results.append(("Random Forest", best, t))

best, sweeps["AdaBoost"], t, _ = search(
    "AdaBoost",
    [{"n_estimators": n, "learning_rate": lr}
     for n in (50, 100, 200, 300) for lr in (0.5, 1.0)],
    lambda n_estimators, learning_rate: AdaBoostClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED),
    key="n_estimators")
results.append(("AdaBoost", best, t))

best, sweeps["XGBoost"], t, _ = search(
    "XGBoost",
    [{"max_depth": d, "learning_rate": lr}
     for d in (4, 5, 6) for lr in (0.03, 0.05, 0.1, 0.2)],
    lambda max_depth, learning_rate: XGBClassifier(
        n_estimators=400, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", scale_pos_weight=spw, random_state=SEED),
    key="learning_rate")
results.append(("XGBoost", best, t))

best, sweeps["LightGBM"], t, _ = search(
    "LightGBM",
    [{"num_leaves": nl, "learning_rate": lr}
     for nl in (15, 31, 63, 127) for lr in (0.05, 0.1)],
    lambda num_leaves, learning_rate: LGBMClassifier(
        n_estimators=400, num_leaves=num_leaves, learning_rate=learning_rate,
        max_depth=-1, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, class_weight="balanced", random_state=SEED, verbosity=-1),
    key="num_leaves")
results.append(("LightGBM", best, t))

best, sweeps["CatBoost"], t, _ = search(
    "CatBoost",
    [{"depth": d, "learning_rate": lr}
     for d in (4, 6, 8) for lr in (0.05, 0.1)],
    lambda depth, learning_rate: CatBoostClassifier(
        iterations=300, depth=depth, learning_rate=learning_rate, l2_leaf_reg=3.0,
        auto_class_weights="Balanced", random_seed=SEED, verbose=False),
    key="depth")
results.append(("CatBoost", best, t))

# ----------------------------------------------------------------- combine
# Deep / foundation / ensemble rows tuned in the notebook (kept as constants).
FIXED = [
    # name, acc, f1, auc, seconds
    ("TabM",          0.9203, 0.9237, 0.9796, 1637.0),
    ("TabPFN",        0.9241, 0.9275, 0.9810, 40.0),
    ("Blend (avg 4)", 0.9226, 0.9258, 0.9799, 60.0),
    ("Stacking (LR)", 0.9232, 0.9266, 0.9807, 61.0),
]

rows = []
for name, best, t in results:
    rows.append({"Model": name, "Accuracy": best["Accuracy"], "F1-Score": best["F1"],
                 "ROC-AUC": best["AUC"], "Seconds": t})
for name, acc, f1, auc, sec in FIXED:
    rows.append({"Model": name, "Accuracy": acc, "F1-Score": f1,
                 "ROC-AUC": auc, "Seconds": sec})

res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
res.round(4).to_csv("model_results.csv", index=False)
Path("notebooks").mkdir(exist_ok=True)
res.round(4).to_csv("notebooks/tuning_results.csv", index=False)
print("\n=== combined leaderboard (best config per model) ===")
print(res.round(4).to_string(index=False))

# ----------------------------------------------------------------- tuning panel
panel = ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]
keylab = {"XGBoost": "learning_rate", "LightGBM": "num_leaves",
          "CatBoost": "depth", "Random Forest": "max_depth"}
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, name in zip(axes.ravel(), panel):
    xs0, ys = zip(*sweeps[name])
    xs = ["None" if v is None else v for v in xs0]
    ax.plot(range(len(xs)), ys, "-o", color=RED, lw=2.2, ms=8)
    bi = int(np.argmax(ys))
    ax.plot(bi, ys[bi], "o", ms=17, mfc="none", mec=DARK, mew=2.4)  # ring = best config
    ax.margins(y=0.25)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_title(name, color=DARK, fontweight="bold")
    ax.set_xlabel(keylab[name]); ax.set_ylabel("CV ROC-AUC")
    ax.grid(alpha=0.25)
fig.suptitle("Hyperparameter search — CV ROC-AUC per configuration",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("figures/fig_tuning_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/fig_tuning_panel.png")

# ----------------------------------------------------------------- GBDT logloss
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                    scale_pos_weight=spw, random_state=SEED)
xgb.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
xgb_ev = xgb.evals_result()

lgb_rec = {}
lgbm = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, class_weight="balanced", random_state=SEED,
                      verbosity=-1)
import lightgbm as lgb
lgbm.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_metric="binary_logloss",
         callbacks=[lgb.record_evaluation(lgb_rec)])

cat = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                         loss_function="Logloss", eval_metric="Logloss",
                         auto_class_weights="Balanced", random_seed=SEED, verbose=False)
cat.fit(Xtr, ytr, eval_set=(Xva, yva))
cat_ev = cat.get_evals_result()

curves = [
    ("XGBoost", xgb_ev["validation_0"]["logloss"], xgb_ev["validation_1"]["logloss"]),
    ("LightGBM", lgb_rec["training"]["binary_logloss"], lgb_rec["valid_1"]["binary_logloss"]),
    ("CatBoost", cat_ev["learn"]["Logloss"], cat_ev["validation"]["Logloss"]),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, (name, tr, va) in zip(axes, curves):
    ax.plot(tr, color=DARK, lw=2, label="train")
    ax.plot(va, color=RED, lw=2, label="validation")
    bi = int(np.argmin(va))
    ax.axvline(bi, color=GREY, ls="--", lw=1.3)
    ax.scatter([bi], [va[bi]], color=RED, zorder=5, s=40)
    ax.set_title(f"{name}  (val min @ {bi})", color=DARK, fontweight="bold")
    ax.set_xlabel("boosting round"); ax.grid(alpha=0.25)
    if ax is axes[0]:
        ax.set_ylabel("logloss"); ax.legend(frameon=False)
fig.suptitle("Gradient boosting — train vs. validation logloss",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/fig_loss_gbdt.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/fig_loss_gbdt.png")
print("DONE make_model_figs")
