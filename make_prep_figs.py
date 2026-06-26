"""Generate data-preprocessing demo figures for the slide deck.
Reads the real raw data and renders before/after illustrations
matching the deck's crimson/black palette.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RED = "#9B1B30"
DARK = "#1f1f1f"
GREY = "#c8c8c8"

plt.rcParams.update({
    "font.size": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

raw = pd.read_csv("data/raw/train.csv")

# ---------------------------------------------------------------- 1. AGE
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

age = raw["age"].astype(float)
median = age.median()

# Before: full range so the lone 383 outlier is obvious
ax[0].hist(age[age <= 100], bins=30, color=DARK, alpha=0.85)
ax[0].scatter([383], [0], color=RED, s=120, zorder=5, clip_on=False)
ax[0].annotate("one value = 383 yr\n(data-entry error)",
               xy=(383, 0), xytext=(250, 220),
               color=RED, fontsize=13, ha="center",
               arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax[0].set_xlim(0, 400)
ax[0].set_title("Before", color=DARK, fontweight="bold")
ax[0].set_xlabel("age (years)")
ax[0].set_ylabel("count")

# After: capped to median
age_after = age.copy()
age_after[age_after > 100] = median
ax[1].hist(age_after, bins=30, color=RED, alpha=0.85)
ax[1].set_xlim(0, 70)
ax[1].set_title(f"After: clip outlier to median ({median:.0f} yr)",
                color=DARK, fontweight="bold")
ax[1].set_xlabel("age (years)")

fig.tight_layout()
fig.savefig("figures/fig_prep_age.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------- 2. ETHNICITY
vc = raw["ethnicity"].value_counts()

def merged_key(v):
    s = str(v).strip().replace("-", " ").lower()
    return "Others" if s == "others" else s.title()

after = raw["ethnicity"].apply(merged_key).value_counts()

fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False)

# Before: highlight the typo duplicates
dup = {"White European", "White-European", "Others", "others"}
cats = vc.index.tolist()[::-1]
vals = vc.values[::-1]
colors = [RED if c in dup else GREY for c in cats]
ax[0].barh(cats, vals, color=colors)
ax[0].set_title("Before  (14 categories)", color=DARK, fontweight="bold")
ax[0].set_xlabel("count")
ax[0].tick_params(axis="y", labelsize=11)

# After
cats2 = after.index.tolist()[::-1]
vals2 = after.values[::-1]
merged_names = {"White European", "Others"}
colors2 = [RED if c in merged_names else GREY for c in cats2]
ax[1].barh(cats2, vals2, color=colors2)
ax[1].set_title("After  (12 categories, typos merged)",
                color=DARK, fontweight="bold")
ax[1].set_xlabel("count")
ax[1].tick_params(axis="y", labelsize=11)

fig.tight_layout()
fig.savefig("figures/fig_prep_ethnicity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- 3. ONE-HOT
sample = raw["relation"].head(6).tolist()
cats = ["Self", "Parent", "Relative", "Health care professional", "Others"]
short = ["Self", "Parent", "Relative", "Health prof.", "Others"]
M = np.array([[1 if r == c else 0 for c in cats] for r in sample])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                       gridspec_kw={"width_ratios": [1, 2.4]})

# Left: the original single column
ax[0].axis("off")
ax[0].set_title("relation\n(one text column)", fontweight="bold", color=DARK)
for i, v in enumerate(sample):
    ax[0].text(0.5, 1 - (i + 1) / 7, v, ha="center", va="center",
               fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="#999999"))

# Right: the one-hot grid
from matplotlib.colors import ListedColormap
ax[1].imshow(M, cmap=ListedColormap(["#f0f0f0", RED]), aspect="auto")
for (i, j), val in np.ndenumerate(M):
    ax[1].text(j, i, val, ha="center", va="center",
               color="white" if val else "#888888", fontsize=12,
               fontweight="bold")
ax[1].set_xticks(range(len(short)))
ax[1].set_xticklabels(short, rotation=30, ha="right", fontsize=11)
ax[1].set_yticks([])
ax[1].set_title("one-hot columns  (0 / 1 — never looks at the label)",
                fontweight="bold", color=DARK)

arr = FancyArrowPatch((0.30, 0.5), (0.36, 0.5),
                      transform=fig.transFigure, mutation_scale=22,
                      color=DARK, lw=2)
fig.patches.append(arr)

fig.tight_layout()
fig.savefig("figures/fig_prep_onehot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved fig_prep_age.png, fig_prep_ethnicity.png, fig_prep_onehot.png")

# ----------------------------------------------------------- 4. DATA OVERVIEW
# A demo of what the raw rows actually look like: the ten AQ-10 yes/no answers,
# age, and the target -- so the audience sees the data, not just a description.
aq = [f"A{i}_Score" for i in range(1, 11)]
demo_cols = aq + ["age", "gender", "austim", "Class/ASD"]
demo = raw[demo_cols].head(6).copy()
demo["age"] = demo["age"].astype(int)
headers = [f"A{i}" for i in range(1, 11)] + ["age", "sex", "fam.", "ASD"]

fig, ax = plt.subplots(figsize=(12, 2.9))
ax.axis("off")
tbl = ax.table(cellText=demo.values, colLabels=headers,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cfcfcf")
    if r == 0:                                   # header row
        cell.set_facecolor(RED); cell.set_text_props(color="white", fontweight="bold")
    elif c == len(headers) - 1:                  # target column
        cell.set_facecolor("#f3d9de")
    elif c <= 9:                                 # the ten AQ-10 answers
        cell.set_facecolor("#f6f6f6")
ax.set_title("Raw sample — ten AQ-10 answers (0/1), age, family history, target",
             color=DARK, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig("figures/fig_data_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig_data_overview.png")
