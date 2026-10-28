# -*- coding: utf-8 -*-
"""
analysis/compute_feature_stats.py
---------------------------------
Compute per-class (Human vs AI) mean, std, and Cohen's d for each of the
16 stylometric features **after** applying the saved StandardScaler.

Outputs:
    saved_models/feature_stats.json
"""

import sys
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train_features.csv"
SCALER_PATH = ROOT_DIR / "saved_models" / "scaler.pkl"
OUTPUT_PATH = ROOT_DIR / "saved_models" / "feature_stats.json"

FEATURE_COLS = [
    "avg_sent_len",
    "std_sent_len",
    "num_sentences",
    "ttr",
    "avg_word_len",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pron_ratio",
    "comma_freq",
    "period_freq",
    "question_freq",
    "avg_paragraph_len",
    "fk_grade",
    "flesh_ease",
]

LABEL_MAP = {0: "Human", 1: "AI"}


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d (pooled-SD variant) between two groups."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def main():
    sep = "=" * 72
    print(sep)
    print("  COMPUTING PER-CLASS FEATURE STATISTICS (SCALED)")
    print(sep)

    # ── Load data & scaler ───────────────────────────────────────────────────
    print(f"\n  Training data : {TRAIN_PATH}")
    print(f"  Scaler        : {SCALER_PATH}")

    df = pd.read_csv(TRAIN_PATH)
    print(f"  Rows loaded   : {len(df):,}")

    scaler = joblib.load(SCALER_PATH)
    print(f"  Scaler features : {scaler.n_features_in_}")

    # ── Extract & scale features ─────────────────────────────────────────────
    X_raw = df[FEATURE_COLS].copy()
    nan_count = X_raw.isna().sum().sum()
    if nan_count > 0:
        print(f"  [WARN] Filling {nan_count} NaN values with column median.")
        X_raw = X_raw.fillna(X_raw.median())

    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        columns=FEATURE_COLS,
        index=df.index,
    )

    labels = df["label"]
    mask_human = labels == 0
    mask_ai = labels == 1
    print(f"  Human samples : {mask_human.sum():,}")
    print(f"  AI samples    : {mask_ai.sum():,}")

    # ── Compute stats ────────────────────────────────────────────────────────
    stats: dict = {"_meta": {
        "description": "Per-class mean/std of scaled features + Cohen's d (Human - AI)",
        "scaler": str(SCALER_PATH),
        "n_human": int(mask_human.sum()),
        "n_ai": int(mask_ai.sum()),
    }}

    for col in FEATURE_COLS:
        human_vals = X_scaled.loc[mask_human, col].values
        ai_vals = X_scaled.loc[mask_ai, col].values
        d = cohens_d(human_vals, ai_vals)

        stats[col] = {
            "human_mean": round(float(np.mean(human_vals)), 6),
            "human_std":  round(float(np.std(human_vals, ddof=1)), 6),
            "ai_mean":    round(float(np.mean(ai_vals)), 6),
            "ai_std":     round(float(np.std(ai_vals, ddof=1)), 6),
            "cohens_d":   round(d, 4),
        }

    # ── Save JSON ────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size:,} bytes)")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FEATURE STATISTICS (scaled)   Cohen's d = (Human − AI) / pooled SD")
    print(sep)

    header = (
        f"  {'Feature':<20}  "
        f"{'Human μ':>9}  {'Human σ':>9}  "
        f"{'AI μ':>9}  {'AI σ':>9}  "
        f"{'Cohen d':>9}  {'|d| Size':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for col in FEATURE_COLS:
        s = stats[col]
        abs_d = abs(s["cohens_d"])
        if abs_d >= 0.8:
            size = "LARGE"
        elif abs_d >= 0.5:
            size = "MEDIUM"
        elif abs_d >= 0.2:
            size = "SMALL"
        else:
            size = "negligible"

        print(
            f"  {col:<20}  "
            f"{s['human_mean']:>9.4f}  {s['human_std']:>9.4f}  "
            f"{s['ai_mean']:>9.4f}  {s['ai_std']:>9.4f}  "
            f"{s['cohens_d']:>+9.4f}  {size:>9}"
        )

    print("  " + "-" * (len(header) - 2))
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
