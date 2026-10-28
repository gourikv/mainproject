# -*- coding: utf-8 -*-
"""
explain.py
----------
Heuristic feature-level explanation for the AI Text Detector.

Given the raw 16 stylometric features for an input text, this module
computes how each feature "pulls" the prediction toward Human or AI
relative to the training-set class averages.

Does NOT modify predict.py.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
STATS_PATH = ROOT_DIR / "saved_models" / "feature_stats.json"

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


def _load_stats() -> dict:
    """Load and cache feature_stats.json."""
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def explain_features(raw_features: dict, scaler) -> List[Dict]:
    """
    Compute per-feature explanation scores.

    Args:
        raw_features: dict with the 16 stylometric feature values (unscaled).
        scaler: a fitted sklearn StandardScaler (same as saved_models/scaler.pkl).

    Returns:
        List of dicts sorted by abs(pull_ai) descending:
            {feature, raw_value, scaled_value, closer_to, pull_ai}

        pull_ai = cohens_d * (scaled_value - midpoint)
            where midpoint = (human_mean + ai_mean) / 2

        Positive pull_ai → feature pushes toward AI.
        Negative pull_ai → feature pushes toward Human.
    """
    stats = _load_stats()

    # Build ordered feature vector and scale it
    feat_list = [raw_features[col] for col in FEATURE_COLS]
    feat_df = pd.DataFrame([feat_list], columns=FEATURE_COLS)
    scaled_arr = scaler.transform(feat_df)[0]  # 1-D array, length 16

    results: List[Dict] = []

    for i, col in enumerate(FEATURE_COLS):
        s = stats[col]
        scaled_val = float(scaled_arr[i])
        human_mean = s["human_mean"]
        ai_mean = s["ai_mean"]
        cohens_d = s["cohens_d"]

        # Pooled SD for this feature (equal class sample sizes: 18370 each)
        pooled_sd = np.sqrt(0.5 * (s["human_std"] ** 2 + s["ai_std"] ** 2))
        sds_from_human = (scaled_val - human_mean) / pooled_sd if pooled_sd > 0 else 0.0
        sds_from_ai = (scaled_val - ai_mean) / pooled_sd if pooled_sd > 0 else 0.0

        # Which class mean is this value closer to?
        dist_human = abs(scaled_val - human_mean)
        dist_ai = abs(scaled_val - ai_mean)
        closer_to = "Human" if dist_human <= dist_ai else "AI"

        # Signed pull score
        #   midpoint sits between the two class means on the scaled axis.
        #   cohens_d is defined as (Human - AI) / pooled_SD, so it is
        #   positive when Human > AI. We want pull_ai > 0 when the feature
        #   pushes the prediction toward AI, so we negate cohens_d:
        #   pull_ai = -cohens_d * (scaled_val - midpoint)
        midpoint = (human_mean + ai_mean) / 2.0
        pull_ai = -cohens_d * (scaled_val - midpoint)

        results.append({
            "feature": col,
            "value": round(float(raw_features[col]), 6),
            "raw_value": round(float(raw_features[col]), 6),
            "scaled_value": round(scaled_val, 4),
            "sds_from_human": round(float(sds_from_human), 3),
            "sds_from_ai": round(float(sds_from_ai), 3),
            "closer_to": closer_to,
            "pull_ai": round(float(pull_ai), 4),
        })

    # Sort by abs(pull_ai) descending
    results.sort(key=lambda r: abs(r["pull_ai"]), reverse=True)
    return results
