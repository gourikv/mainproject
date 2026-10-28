# -*- coding: utf-8 -*-
"""
fit_scaler.py
-------------
Fits a scikit-learn StandardScaler on the 16 stylometric features extracted
from the training split (data/processed/train_features.csv) and saves the fitted
scaler to saved_models/scaler.pkl.
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "train_features.csv"
OUTPUT_DIR = ROOT_DIR / "saved_models"
SCALER_PATH = OUTPUT_DIR / "scaler.pkl"

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


def main():
    print("=" * 70)
    print("FITTING STANDARD SCALER ON TRAINING STYLOMETRIC FEATURES")
    print("=" * 70)

    # 1. Load training features
    print(f"Loading features from: {TRAIN_FEATURES_PATH}")
    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(f"File not found: {TRAIN_FEATURES_PATH}")

    df = pd.read_csv(TRAIN_FEATURES_PATH)
    print(f"Loaded {len(df):,} rows.")

    # 2. Select exact 16 columns
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    X_train = df[FEATURE_COLS]
    print(f"Selected {len(FEATURE_COLS)} feature columns in exact order.")

    # Check for any NaNs or infinities
    nan_count = X_train.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values; filling with median.")
        X_train = X_train.fillna(X_train.median())

    # 3. Fit StandardScaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 4. Save fitted scaler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved fitted scaler to: {SCALER_PATH} (size: {SCALER_PATH.stat().st_size:,} bytes)")

    # 5. Print mean_ and scale_ arrays
    print("\n" + "=" * 70)
    print("SCALER PARAMETERS (mean_ and scale_)")
    print("=" * 70)

    print("\n--- Raw scaler.mean_ ---")
    print(scaler.mean_)

    print("\n--- Raw scaler.scale_ (std dev) ---")
    print(scaler.scale_)

    print("\n--- Detailed per-feature summary ---")
    header = f"{'Feature':<20} | {'Mean (mean_)':>15} | {'Std Dev (scale_)':>18}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for col, mean_val, scale_val in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        print(f"{col:<20} | {mean_val:>15.6f} | {scale_val:>18.6f}")
    print("-" * len(header))

    print("\nScaler fitting and verification complete.")


if __name__ == "__main__":
    main()
