# -*- coding: utf-8 -*-
"""
predict.py
----------
End-to-end inference script using the trained HybridClassifier (RoBERTa-base +
16 stylometric features) and the fitted StandardScaler.
"""

import sys
import io
from pathlib import Path
from typing import Tuple, Dict

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel
import spacy
import joblib
import numpy as np
import pandas as pd

# Import feature extractor from src/features.py
from src.features import extract_features

ROOT_DIR = Path(__file__).resolve().parent
SCALER_PATH = ROOT_DIR / "saved_models" / "scaler.pkl"
MODEL_PATH = ROOT_DIR / "saved_models" / "hybrid_model.pt"

# Exact 16 feature columns in order
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------
# 1. Model Architecture (matching verify_models.py)
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


# -----------------------------------------------------------------------------
# 2. Load spaCy model
# -----------------------------------------------------------------------------
print("Loading spaCy model (en_core_web_sm)...")
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("  en_core_web_sm not found, downloading...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
print("spaCy model loaded.")


# -----------------------------------------------------------------------------
# 3. Load Scaler
# -----------------------------------------------------------------------------
print(f"Loading scaler from: {SCALER_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Run fit_scaler.py first.")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded.")


# -----------------------------------------------------------------------------
# 4. Load Hybrid Model & Tokenizer
# -----------------------------------------------------------------------------
print(f"Loading hybrid model weights from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
hybrid_model = HybridClassifier(n_stylo_features=len(FEATURE_COLS)).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
hybrid_model.load_state_dict(state_dict)
hybrid_model.eval()
print("Hybrid model loaded successfully.\n")


# -----------------------------------------------------------------------------
# 5. Predict Function
# -----------------------------------------------------------------------------
def predict(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Run end-to-end prediction on raw input text.

    Args:
        text: Raw text string to classify.

    Returns:
        tuple: (label: str, probabilities: dict)
               label: 'Human-written' or 'AI-generated'
               probabilities: {'Human-written': float, 'AI-generated': float}
    """
    # 1. Run through spaCy
    doc = nlp(text)

    # 2. Extract 16 stylometric features
    feat_dict = extract_features(text, doc)

    # 3. Convert to list in exact order
    feat_list = [feat_dict[col] for col in FEATURE_COLS]

    # 4. Scale features using fitted StandardScaler (using DataFrame to preserve feature names)
    feat_df = pd.DataFrame([feat_list], columns=FEATURE_COLS)
    scaled_feats = scaler.transform(feat_df)
    stylo_tensor = torch.tensor(scaled_feats, dtype=torch.float).to(device)

    # 5. Tokenize text with RoBERTa tokenizer
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    ).to(device)

    # 6. Model forward pass
    with torch.no_grad():
        logits = hybrid_model(enc["input_ids"], enc["attention_mask"], stylo_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    label = "AI-generated" if pred_idx == 1 else "Human-written"
    prob_dict = {
        "Human-written": float(probs[0].item()),
        "AI-generated": float(probs[1].item()),
    }

    return label, prob_dict


# -----------------------------------------------------------------------------
# 6. Test on Sample Texts
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    human_sample = (
        "honestly idk why people keep hyping up that new restaurant downtown lol. "
        "me and my buddy waited like 45 mins in the freezing rain just for soggy fries "
        "and a burger that tasted completely burnt. totally not worth the 30 bucks!! "
        "save your cash and just grab some tacos down the street instead imo."
    )

    ai_sample = (
        "Sure! When you use your computer, it stores files on your hard drive. "
        "Sometimes, when you save a file, your computer might not have enough space next to the other files "
        "to store the whole file in one piece. Instead, it has to save part of the file in one place "
        "and the rest in another place. This is called fragmentation. A disk defragmenter fixes this by "
        "reorganizing those pieces back together so your computer can locate them much faster and run smoothly. "
        "I hope this helps! Let me know if you have any questions."
    )

    test_samples = [
        ("Casual / Human-sounding Text", human_sample),
        ("Formal / AI-sounding Text", ai_sample),
    ]

    print("=" * 75)
    print("RUNNING INFERENCE ON SAMPLE TEXTS")
    print("=" * 75)

    for desc, sample in test_samples:
        print(f"\n[{desc}]")
        print(f"Text: \"{sample}\"")
        label, probs = predict(sample)
        print(f"-> Prediction    : {label}")
        print(f"-> Probabilities : Human-written: {probs['Human-written']*100:.2f}% | AI-generated: {probs['AI-generated']*100:.2f}%")

    print("\n" + "=" * 75)
    print("INFERENCE VERIFICATION FINISHED")
    print("=" * 75)
