"""
Verification script — confirms both saved models load correctly
and can run a prediction on sample text.

Run this from your project root:
    python verify_models.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

FEATURE_COLS = ['avg_sent_len', 'std_sent_len', 'num_sentences', 'ttr', 'avg_word_len',
                'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pron_ratio',
                'comma_freq', 'period_freq', 'question_freq', 'avg_paragraph_len',
                'fk_grade', 'flesh_ease']

# ---------------------------------------------------------------
# 1. Define the exact same architecture used during training
# ---------------------------------------------------------------
class HybridClassifier(nn.Module):
    def __init__(self, n_stylo_features, hidden_dim=128, n_classes=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.stylo_fc = nn.Sequential(
            nn.Linear(n_stylo_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        combined_dim = self.roberta.config.hidden_size + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask, stylo_feats):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = roberta_out.last_hidden_state[:, 0, :]
        stylo_out = self.stylo_fc(stylo_feats)
        combined = torch.cat([cls_embedding, stylo_out], dim=1)
        logits = self.classifier(combined)
        return logits


# ---------------------------------------------------------------
# 2. Load the Hybrid model
# ---------------------------------------------------------------
print("=" * 60)
print("LOADING HYBRID MODEL")
print("=" * 60)

try:
    hybrid_model = HybridClassifier(n_stylo_features=len(FEATURE_COLS)).to(device)
    state_dict = torch.load("saved_models/hybrid_model.pt", map_location=device)
    hybrid_model.load_state_dict(state_dict)
    hybrid_model.eval()
    print("✅ Hybrid model loaded successfully — weights match architecture.\n")
except Exception as e:
    print(f"❌ Failed to load hybrid model: {e}\n")
    hybrid_model = None

# ---------------------------------------------------------------
# 3. Load the RoBERTa baseline
# ---------------------------------------------------------------
print("=" * 60)
print("LOADING ROBERTA BASELINE")
print("=" * 60)

try:
    baseline_tokenizer = AutoTokenizer.from_pretrained("saved_models/roberta_baseline_final")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_models/roberta_baseline_final"
    ).to(device)
    baseline_model.eval()
    print("✅ RoBERTa baseline loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load RoBERTa baseline: {e}\n")
    baseline_model = None
    baseline_tokenizer = None

# ---------------------------------------------------------------
# 4. Run a test prediction through the Hybrid model
# ---------------------------------------------------------------
if hybrid_model is not None:
    print("=" * 60)
    print("TEST PREDICTION — HYBRID MODEL")
    print("=" * 60)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    sample_text = "The quick brown fox jumps over the lazy dog. This is a simple test sentence."

    enc = tokenizer(sample_text, truncation=True, padding="max_length",
                     max_length=256, return_tensors="pt").to(device)

    # NOTE: using zeros as a placeholder for the 16 stylometric features here.
    # This is ONLY for verifying the model loads and runs without shape errors —
    # real predictions need properly scaled stylometric features (see note below).
    dummy_stylo_feats = torch.zeros((1, len(FEATURE_COLS)), dtype=torch.float).to(device)

    with torch.no_grad():
        logits = hybrid_model(enc["input_ids"], enc["attention_mask"], dummy_stylo_feats)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()

    label = "AI-generated" if pred == 1 else "Human-written"
    print(f"Sample text: \"{sample_text}\"")
    print(f"Raw logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    print(f"Prediction: {label}")
    print("\n⚠️  NOTE: stylometric features were zeroed out for this test — this only")
    print("   confirms the model architecture and weights load correctly, NOT that")
    print("   predictions are meaningful yet. Real inference needs the saved scaler.\n")

# ---------------------------------------------------------------
# 5. Run a test prediction through the RoBERTa baseline
# ---------------------------------------------------------------
if baseline_model is not None:
    print("=" * 60)
    print("TEST PREDICTION — ROBERTA BASELINE")
    print("=" * 60)

    sample_text = "The quick brown fox jumps over the lazy dog. This is a simple test sentence."
    enc = baseline_tokenizer(sample_text, truncation=True, padding="max_length",
                              max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        output = baseline_model(**enc)
        probs = torch.softmax(output.logits, dim=1)
        pred = torch.argmax(output.logits, dim=1).item()

    label = "AI-generated" if pred == 1 else "Human-written"
    print(f"Sample text: \"{sample_text}\"")
    print(f"Raw logits: {output.logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    print(f"Prediction: {label}\n")

print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
