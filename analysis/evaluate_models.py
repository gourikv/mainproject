# -*- coding: utf-8 -*-
"""
evaluate_models.py
------------------
Evaluates both the RoBERTa baseline and the HybridClassifier on the full test set
(data/processed/test.csv and data/processed/test_features.csv).

Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrices for both models,
prints a side-by-side comparison table, and saves raw predictions to analysis/eval_results.csv.
"""

import sys
import io
from pathlib import Path
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CSV_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
TEST_FEATS_PATH = ROOT_DIR / "data" / "processed" / "test_features.csv"
SCALER_PATH = ROOT_DIR / "saved_models" / "scaler.pkl"
BASELINE_DIR = ROOT_DIR / "saved_models" / "roberta_baseline_final"
HYBRID_MODEL_PATH = ROOT_DIR / "saved_models" / "hybrid_model.pt"
OUTPUT_CSV_PATH = ROOT_DIR / "analysis" / "eval_results.csv"

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

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    import os
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)
    print(f"Configured PyTorch CPU threads: {cpu_cores}")


# -----------------------------------------------------------------------------
# 1. Architecture definition for HybridClassifier
# -----------------------------------------------------------------------------
class HybridClassifier(nn.Module):
    def __init__(self, n_stylo_features: int, hidden_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.stylo_fc = nn.Sequential(
            nn.Linear(n_stylo_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        combined_dim = self.roberta.config.hidden_size + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, stylo_feats):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = roberta_out.last_hidden_state[:, 0, :]
        stylo_out = self.stylo_fc(stylo_feats)
        combined = torch.cat([cls_embedding, stylo_out], dim=1)
        logits = self.classifier(combined)
        return logits


def main():
    print("=" * 75)
    print("MODEL EVALUATION PIPELINE (ROBERTA BASELINE vs. HYBRID MODEL)")
    print("=" * 75)
    print(f"Hardware device in use: {device}")

    # 1. Load test data
    print(f"\n[1/6] Loading test dataset...")
    if not TEST_CSV_PATH.exists() or not TEST_FEATS_PATH.exists():
        raise FileNotFoundError(f"Missing test CSV files in {ROOT_DIR / 'data' / 'processed'}")

    test_df = pd.read_csv(TEST_CSV_PATH)
    feats_df = pd.read_csv(TEST_FEATS_PATH)

    print(f"  test.csv rows          : {len(test_df):,}")
    print(f"  test_features.csv rows : {len(feats_df):,}")
    assert len(test_df) == len(feats_df), "Mismatch in row counts between test.csv and test_features.csv"

    # Randomly sample 1000 rows with fixed random_state=42
    sample_size = min(1000, len(test_df))
    sample_indices = test_df.sample(n=sample_size, random_state=42).index
    test_df = test_df.loc[sample_indices].reset_index(drop=True)
    feats_df = feats_df.loc[sample_indices].reset_index(drop=True)
    print(f"  Sampled for evaluation : {len(test_df):,} rows (random_state=42)")

    # Fill any missing text or NaN features
    test_texts = test_df["text"].fillna("").tolist()
    y_true = test_df["label"].values

    # 2. Load and scale stylometric features
    print(f"\n[2/6] Loading scaler and scaling {len(FEATURE_COLS)} stylometric features...")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    X_feats = feats_df[FEATURE_COLS]
    if X_feats.isna().sum().sum() > 0:
        X_feats = X_feats.fillna(X_feats.median())
    scaled_feats = scaler.transform(X_feats)
    all_stylo_tensors = torch.tensor(scaled_feats, dtype=torch.float)
    print(f"  Scaled features shape: {scaled_feats.shape}")

    # 3. Load RoBERTa Baseline
    print(f"\n[3/6] Loading RoBERTa baseline model from: {BASELINE_DIR}")
    baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_DIR)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_DIR).to(device)
    baseline_model.eval()
    print("  Baseline loaded successfully.")

    # 4. Load Hybrid Model
    print(f"\n[4/6] Loading Hybrid model from: {HYBRID_MODEL_PATH}")
    hybrid_model = HybridClassifier(n_stylo_features=len(FEATURE_COLS)).to(device)
    hybrid_state = torch.load(HYBRID_MODEL_PATH, map_location=device)
    hybrid_model.load_state_dict(hybrid_state)
    hybrid_model.eval()
    print("  Hybrid model loaded successfully.")

    # 5. Run inference in batches
    print(f"\n[5/6] Running batched inference (batch size = {BATCH_SIZE})...")
    total_samples = len(test_df)
    baseline_preds = []
    baseline_probs_human = []
    baseline_probs_ai = []

    hybrid_preds = []
    hybrid_probs_human = []
    hybrid_probs_ai = []

    start_time = time.time()

    with torch.no_grad():
        for start_idx in tqdm(range(0, total_samples, BATCH_SIZE), desc="Evaluating Batches", ncols=85):
            end_idx = min(start_idx + BATCH_SIZE, total_samples)
            batch_texts = test_texts[start_idx:end_idx]
            batch_stylo = all_stylo_tensors[start_idx:end_idx].to(device)

            # Tokenize once for both models (dynamic padding to max length in batch)
            enc = baseline_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            # Baseline inference
            out_base = baseline_model(**enc)
            probs_base = torch.softmax(out_base.logits, dim=1)
            preds_base = torch.argmax(probs_base, dim=1)

            baseline_preds.extend(preds_base.cpu().numpy().tolist())
            baseline_probs_human.extend(probs_base[:, 0].cpu().numpy().tolist())
            baseline_probs_ai.extend(probs_base[:, 1].cpu().numpy().tolist())

            # Hybrid inference
            out_hybrid = hybrid_model(enc["input_ids"], enc["attention_mask"], batch_stylo)
            probs_hybrid = torch.softmax(out_hybrid, dim=1)
            preds_hybrid = torch.argmax(probs_hybrid, dim=1)

            hybrid_preds.extend(preds_hybrid.cpu().numpy().tolist())
            hybrid_probs_human.extend(probs_hybrid[:, 0].cpu().numpy().tolist())
            hybrid_probs_ai.extend(probs_hybrid[:, 1].cpu().numpy().tolist())

    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.1f}s ({total_samples / elapsed:.1f} samples/sec).")

    # Convert to numpy arrays
    y_pred_base = np.array(baseline_preds)
    y_pred_hybrid = np.array(hybrid_preds)

    # 6. Compute metrics
    print(f"\n[6/6] Computing evaluation metrics and confusion matrices...")

    # Baseline metrics
    acc_base = accuracy_score(y_true, y_pred_base)
    prec_base = precision_score(y_true, y_pred_base, zero_division=0)
    rec_base = recall_score(y_true, y_pred_base, zero_division=0)
    f1_base = f1_score(y_true, y_pred_base, zero_division=0)
    cm_base = confusion_matrix(y_true, y_pred_base)

    # Hybrid metrics
    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
    prec_hybrid = precision_score(y_true, y_pred_hybrid, zero_division=0)
    rec_hybrid = recall_score(y_true, y_pred_hybrid, zero_division=0)
    f1_hybrid = f1_score(y_true, y_pred_hybrid, zero_division=0)
    cm_hybrid = confusion_matrix(y_true, y_pred_hybrid)

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 75)
    print("                MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 75)
    header = f"{'Metric':<20} | {'RoBERTa Baseline':>18} | {'Hybrid Model':>18} | {'Delta':>10}"
    print(header)
    print("-" * len(header))

    metrics_list = [
        ("Accuracy", acc_base, acc_hybrid),
        ("Precision (AI)", prec_base, prec_hybrid),
        ("Recall (AI)", rec_base, rec_hybrid),
        ("F1-Score (AI)", f1_base, f1_hybrid),
    ]

    for name, m_base, m_hyb in metrics_list:
        delta = m_hyb - m_base
        delta_str = f"{delta:+.4f}"
        print(f"{name:<20} | {m_base:>18.4f} | {m_hyb:>18.4f} | {delta_str:>10}")
    print("-" * len(header))

    # Print Confusion Matrices
    print("\n" + "=" * 75)
    print("CONFUSION MATRICES")
    print("=" * 75)
    print("Labels: 0 = Human-written, 1 = AI-generated (ChatGPT)\n")

    print("--- RoBERTa Baseline Confusion Matrix ---")
    print(f"                  Pred: Human (0)     Pred: AI (1)")
    print(f"Actual: Human (0)     {cm_base[0, 0]:>11,d}      {cm_base[0, 1]:>11,d}")
    print(f"Actual: AI    (1)     {cm_base[1, 0]:>11,d}      {cm_base[1, 1]:>11,d}")

    print("\n--- Hybrid Model Confusion Matrix ---")
    print(f"                  Pred: Human (0)     Pred: AI (1)")
    print(f"Actual: Human (0)     {cm_hybrid[0, 0]:>11,d}      {cm_hybrid[0, 1]:>11,d}")
    print(f"Actual: AI    (1)     {cm_hybrid[1, 0]:>11,d}      {cm_hybrid[1, 1]:>11,d}")

    # 7. Save raw predictions to analysis/eval_results.csv
    eval_results_df = pd.DataFrame({
        "text": test_df["text"],
        "true_label": y_true,
        "source": test_df.get("source", ""),
        "baseline_pred": y_pred_base,
        "baseline_prob_human": baseline_probs_human,
        "baseline_prob_ai": baseline_probs_ai,
        "hybrid_pred": y_pred_hybrid,
        "hybrid_prob_human": hybrid_probs_human,
        "hybrid_prob_ai": hybrid_probs_ai,
    })

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved evaluation predictions to: {OUTPUT_CSV_PATH}")
    print(f"  Rows saved: {len(eval_results_df):,} | File size: {OUTPUT_CSV_PATH.stat().st_size:,} bytes")
    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
