# -*- coding: utf-8 -*-
"""
features.py
-----------
Step 3: Stylometric feature extraction for HC3 processed splits.

Features per text sample:
  Sentence-level : avg_sent_len, std_sent_len, num_sentences
  Lexical        : ttr, avg_word_len
  POS (spaCy)    : noun_ratio, verb_ratio, adj_ratio, adv_ratio, pron_ratio
  Punctuation    : comma_freq, period_freq, question_freq, avg_paragraph_len
  Readability    : fk_grade, flesh_ease

Outputs: data/processed/{train,val,test}_features.csv
"""


import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import spacy
import textstat
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
PROCESSED = ROOT_DIR / "data" / "processed"


# ── Load spaCy ───────────────────────────────────────────────────────────────
def section(title: str, width: int = 68) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"

print(section("Loading spaCy model"))
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print(f"  Loaded: en_core_web_sm  pipes={nlp.pipe_names}")
except OSError:
    print("  en_core_web_sm not found. Downloading...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print(f"  Loaded after download. pipes={nlp.pipe_names}")


# ── Feature extractor ────────────────────────────────────────────────────────
def extract_features(text: str, doc) -> dict:
    """Extract all stylometric features from a text string and its spaCy Doc."""
    tokens     = [t for t in doc if not t.is_space]
    words      = [t for t in tokens if t.is_alpha]
    sents      = list(doc.sents)
    n_tokens   = len(tokens)
    n_words    = len(words)

    # ── Sentence-level ────────────────────────────────────────────────────────
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]
    avg_sent_len = float(np.mean(sent_lens))  if sent_lens else 0.0
    std_sent_len = float(np.std(sent_lens))   if len(sent_lens) > 1 else 0.0
    num_sentences = len(sents)

    # ── Lexical diversity ─────────────────────────────────────────────────────
    lower_words = [w.lower_ for w in words]
    ttr         = len(set(lower_words)) / n_words if n_words > 0 else 0.0
    avg_word_len = float(np.mean([len(w) for w in lower_words])) if lower_words else 0.0

    # ── POS ratios ────────────────────────────────────────────────────────────
    if n_tokens > 0:
        pos_counts = {}
        for t in tokens:
            pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
        noun_ratio = pos_counts.get("NOUN", 0) / n_tokens
        verb_ratio = pos_counts.get("VERB", 0) / n_tokens
        adj_ratio  = pos_counts.get("ADJ",  0) / n_tokens
        adv_ratio  = pos_counts.get("ADV",  0) / n_tokens
        pron_ratio = pos_counts.get("PRON", 0) / n_tokens
    else:
        noun_ratio = verb_ratio = adj_ratio = adv_ratio = pron_ratio = 0.0

    # ── Punctuation & structure ───────────────────────────────────────────────
    n_chars  = max(len(text), 1)
    comma_freq    = text.count(",")    / n_chars
    period_freq   = text.count(".")    / n_chars
    question_freq = text.count("?")    / n_chars

    # Paragraph length: split on double-newline; if none, treat whole text as 1 para
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    avg_paragraph_len = float(np.mean([len(p.split()) for p in paragraphs])) \
                        if paragraphs else float(n_words)

    # ── Readability ───────────────────────────────────────────────────────────
    try:
        fk_grade   = textstat.flesch_kincaid_grade(text)
        flesh_ease = textstat.flesch_reading_ease(text)
    except Exception:
        fk_grade = flesh_ease = 0.0

    return {
        "avg_sent_len":      avg_sent_len,
        "std_sent_len":      std_sent_len,
        "num_sentences":     num_sentences,
        "ttr":               ttr,
        "avg_word_len":      avg_word_len,
        "noun_ratio":        noun_ratio,
        "verb_ratio":        verb_ratio,
        "adj_ratio":         adj_ratio,
        "adv_ratio":         adv_ratio,
        "pron_ratio":        pron_ratio,
        "comma_freq":        comma_freq,
        "period_freq":       period_freq,
        "question_freq":     question_freq,
        "avg_paragraph_len": avg_paragraph_len,
        "fk_grade":          fk_grade,
        "flesh_ease":        flesh_ease,
    }


# ── Process a split ──────────────────────────────────────────────────────────
def process_split(name: str) -> pd.DataFrame:
    csv_path = PROCESSED / f"{name}.csv"
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} not found — skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"\n  Processing '{name}': {len(df):,} rows ...")

    texts = df["text"].fillna("").tolist()
    feature_rows = []

    # Batch spaCy inference for speed (batch_size=128)
    for doc in tqdm(
        nlp.pipe(texts, batch_size=128),
        total=len(texts),
        desc=f"  [{name}] spaCy",
        ncols=80,
    ):
        text = doc.text
        feature_rows.append(extract_features(text, doc))

    feat_df = pd.DataFrame(feature_rows)
    result  = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out_path = PROCESSED / f"{name}_features.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved -> {out_path}  ({len(result):,} rows x {len(result.columns)} cols)")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(section("Extracting Stylometric Features"))

    results = {}
    for split in ["train", "val", "test"]:
        results[split] = process_split(split)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(section("Feature File Summary"))
    feature_cols = [
        "avg_sent_len", "std_sent_len", "num_sentences",
        "ttr", "avg_word_len",
        "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio",
        "comma_freq", "period_freq", "question_freq", "avg_paragraph_len",
        "fk_grade", "flesh_ease",
    ]
    print(f"\n  Total features : {len(feature_cols)}")
    print(f"  Feature names  : {feature_cols}")

    for name, df in results.items():
        if df.empty:
            continue
        print(f"\n  -- {name} split ({len(df):,} rows) --")
        print(f"  Columns: {list(df.columns)}")
        print(df[feature_cols].describe().round(3).to_string())

    print("\n  Feature extraction complete!")

