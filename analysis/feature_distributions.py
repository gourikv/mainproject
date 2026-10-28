# -*- coding: utf-8 -*-
"""
feature_distributions.py
------------------------
Plots distribution of each stylometric feature split by label (human vs ChatGPT),
runs Welch's t-test + Cohen's d to rank features by discriminative power,
and prints the top-5 most separating features.

Outputs: analysis/plots/{feature_name}.png
"""

import sys
import io
# Bootstrap: ensure AppData site-packages are on path (Python 3.14 on Windows)
_APPDATA_PKGS = r"C:\Users\aaa\AppData\Roaming\Python\Python314\site-packages"
if _APPDATA_PKGS not in sys.path:
    sys.path.insert(0, _APPDATA_PKGS)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
PROCESSED = ROOT_DIR / "data" / "processed"
PLOTS_DIR = ROOT_DIR / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "avg_sent_len", "std_sent_len", "num_sentences",
    "ttr", "avg_word_len",
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio",
    "comma_freq", "period_freq", "question_freq", "avg_paragraph_len",
    "fk_grade", "flesh_ease",
]

LABEL_NAMES = {0: "Human", 1: "ChatGPT"}
COLORS      = {0: "#4C72B0", 1: "#DD8452"}   # blue = human, orange = chatgpt


def section(title: str, width: int = 68) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 +
                          (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


# ── Load train_features (largest split for most stable distributions) ────────
print(section("Loading feature data"))
feat_path = PROCESSED / "train_features.csv"
if not feat_path.exists():
    print(f"  [ERROR] {feat_path} not found. Run src/features.py first.")
    sys.exit(1)

df = pd.read_csv(feat_path)
print(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")

human   = df[df["label"] == 0]
chatgpt = df[df["label"] == 1]
print(f"  Human  : {len(human):,}  ChatGPT: {len(chatgpt):,}")

# ── Plot each feature ────────────────────────────────────────────────────────
print(section("Generating distribution plots"))

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

saved_plots = []
for feat in FEATURE_COLS:
    if feat not in df.columns:
        print(f"  [SKIP] {feat} not in DataFrame")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Feature: {feat}", fontsize=14, fontweight="bold")

    # Clip extreme outliers for plotting readability (1st–99th pct)
    lo = df[feat].quantile(0.01)
    hi = df[feat].quantile(0.99)
    plot_df = df[(df[feat] >= lo) & (df[feat] <= hi)].copy()

    # Left: overlapping histogram
    ax = axes[0]
    for lbl, color in COLORS.items():
        subset = plot_df[plot_df["label"] == lbl][feat].dropna()
        ax.hist(subset, bins=50, alpha=0.55, color=color,
                label=LABEL_NAMES[lbl], density=True)
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Histogram (density)")

    # Right: side-by-side boxplot
    ax2 = axes[1]
    bp_data = [
        plot_df[plot_df["label"] == lbl][feat].dropna().values
        for lbl in [0, 1]
    ]
    bp = ax2.boxplot(bp_data, patch_artist=True, notch=True,
                     tick_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax2.set_ylabel(feat)
    ax2.set_title("Box plot")

    out = PLOTS_DIR / f"{feat}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    saved_plots.append(out)
    print(f"  Saved: {out.name}")

# ── Statistical ranking ──────────────────────────────────────────────────────
print(section("Statistical ranking of features"))

records = []
for feat in FEATURE_COLS:
    if feat not in df.columns:
        continue
    a = human[feat].dropna().values
    b = chatgpt[feat].dropna().values
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    d             = cohens_d(a, b)
    records.append({
        "feature":      feat,
        "mean_human":   a.mean(),
        "mean_chatgpt": b.mean(),
        "t_stat":       t_stat,
        "p_value":      p_val,
        "cohens_d":     d,
        "abs_d":        abs(d),
    })

rank_df = pd.DataFrame(records).sort_values("abs_d", ascending=False).reset_index(drop=True)

print("\n  All features ranked by |Cohen's d| (effect size):\n")
print(f"  {'Rank':<5} {'Feature':<22} {'Mean Human':>12} {'Mean GPT':>10}"
      f" {'|Cohen d|':>10} {'p-value':>12}")
print("  " + "-" * 75)
for i, row in rank_df.iterrows():
    sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else "*")
    print(f"  {i+1:<5} {row['feature']:<22} {row['mean_human']:>12.4f} "
          f"{row['mean_chatgpt']:>10.4f} {row['abs_d']:>10.4f} "
          f"{row['p_value']:>12.2e} {sig}")

print("\n  TOP 5 MOST DISCRIMINATIVE FEATURES:")
print("  " + "=" * 55)
for i, row in rank_df.head(5).iterrows():
    direction = "Human > ChatGPT" if row["mean_human"] > row["mean_chatgpt"] else "ChatGPT > Human"
    print(f"  #{i+1}  {row['feature']:<22}  |d|={row['abs_d']:.4f}  ({direction})")

# Save ranking
rank_csv = ROOT_DIR / "analysis" / "feature_ranking.csv"
rank_df.to_csv(rank_csv, index=False)
print(f"\n  Full ranking saved -> {rank_csv}")
print(f"  Plots saved to     -> {PLOTS_DIR}")
print("\n  Analysis complete!")
