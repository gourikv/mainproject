# -*- coding: utf-8 -*-
"""
preprocess.py
-------------
Step 2: Data cleaning and preprocessing pipeline for HC3.

Actions performed:
  1. Load the 5 JSONL domain files from HuggingFace Hub (skips all.jsonl)
  2. Reshape to flat format: [text, label, source]
     - label 0 = human answer, label 1 = ChatGPT answer
     - each element of the answer lists becomes its own row (explode)
  3. Clean text (HTML, markdown, whitespace)
  4. Drop rows with text < 20 chars and exact duplicates
  5. Balance classes by downsampling the majority
  6. Stratified 70/15/15 train/val/test split
  7. Save to data/processed/train.csv, val.csv, test.csv
  8. Print summary
"""

import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

REPO_ID     = "Hello-SimpleAI/HC3"
RANDOM_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def section(title: str, width: int = 68) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def clean_text(text: str) -> str:
    """Strip HTML tags, markdown artifacts, and normalise whitespace."""
    if not isinstance(text, str):
        text = str(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove markdown links  [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove markdown bold/italic markers
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    # Remove markdown headers (#, ##, ...)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove code fences
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    # Collapse whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_summary(df: pd.DataFrame, name: str) -> str:
    """Return a compact summary string for a data split."""
    counts  = df["label"].value_counts().sort_index()
    avg_len = df["text"].str.len().mean()
    h_count = counts.get(0, 0)
    c_count = counts.get(1, 0)
    return (
        f"  {name:<8}  rows={len(df):>6,}   "
        f"human={h_count:>6,}  chatgpt={c_count:>6,}   "
        f"avg_len={avg_len:>7.1f} chars"
    )


# ── 1. Load dataset from HuggingFace ────────────────────────────────────────
print(section("1. Loading HC3 from HuggingFace Hub (domain files only)"))

# 'all.jsonl' is a concatenation of the 5 domain files — loading it alongside
# the individual files would double every record and create ~91k duplicates.
# We explicitly whitelist only the domain-specific files.
DOMAIN_FILES = [
    "finance.jsonl",
    "medicine.jsonl",
    "open_qa.jsonl",
    "reddit_eli5.jsonl",
    "wiki_csai.jsonl",
]

all_repo_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
available = [f for f in all_repo_files if f in DOMAIN_FILES]
skipped   = [f for f in all_repo_files if f.endswith(".jsonl") and f not in DOMAIN_FILES]

print(f"  Loading  : {available}")
print(f"  Skipping : {skipped}")

# Load each domain file individually to avoid Arrow schema mismatch.
# (finance.jsonl has an extra 'index' column; bulk load_dataset fails on it.)
frames = []
for fname in available:
    local_path = hf_hub_download(
        repo_id=REPO_ID, filename=fname, repo_type="dataset"
    )
    df_part = pd.read_json(local_path, lines=True)
    # Ensure 'source' column exists (some files already have it)
    if "source" not in df_part.columns:
        df_part["source"] = fname.replace(".jsonl", "")
    frames.append(df_part)
    print(f"    {fname:<25} {len(df_part):>6,} rows  cols={list(df_part.columns)}")

raw_df = pd.concat(frames, ignore_index=True)
print(f"  Total loaded: {len(raw_df):,} rows, columns: {list(raw_df.columns)}")


# ── 2. Reshape to flat format ────────────────────────────────────────────────
print(section("2. Reshaping to flat [text, label, source] format"))

records = []
for _, row in raw_df.iterrows():
    src = row.get("source", "unknown")

    # Human answers (label 0)
    human = row.get("human_answers", None)
    if human is None:
        human = []
    if isinstance(human, str):
        human = [human]
    else:
        human = list(human)  # converts numpy arrays / lists safely

    for ans in human:
        ans = str(ans).strip() if ans is not None else ""
        if ans:
            records.append({"text": ans, "label": 0, "source": src})

    # ChatGPT answers (label 1)
    chatgpt = row.get("chatgpt_answers", None)
    if chatgpt is None:
        chatgpt = []
    if isinstance(chatgpt, str):
        chatgpt = [chatgpt]
    else:
        chatgpt = list(chatgpt)

    for ans in chatgpt:
        ans = str(ans).strip() if ans is not None else ""
        if ans:
            records.append({"text": ans, "label": 1, "source": src})


flat_df = pd.DataFrame(records)
print(f"  Exploded rows : {len(flat_df):,}")
print(f"  Label counts  : {flat_df['label'].value_counts().to_dict()}")


# ── 3. Clean text ────────────────────────────────────────────────────────────
print(section("3. Cleaning text"))

flat_df["text"] = flat_df["text"].apply(clean_text)

before = len(flat_df)
# Drop too-short texts
flat_df = flat_df[flat_df["text"].str.len() >= 20].copy()
after_short = len(flat_df)
print(f"  Removed (< 20 chars) : {before - after_short:,} rows")

# Drop exact duplicates
before_dup = len(flat_df)
dup_count  = flat_df.duplicated(subset="text").sum()
flat_df    = flat_df.drop_duplicates(subset="text").copy()
after_dup  = len(flat_df)
print(f"  Removed (duplicates) : {before_dup - after_dup:,} rows  ({dup_count:,} exact text duplicates)")
print(f"  Remaining            : {after_dup:,} rows")
print(f"  Label counts         : {flat_df['label'].value_counts().to_dict()}")
print(f"  Unique sources       : {sorted(flat_df['source'].dropna().unique().tolist())}")


# ── 4. Balance classes ───────────────────────────────────────────────────────
print(section("4. Balancing classes"))

counts = flat_df["label"].value_counts()
min_count = counts.min()
maj_label = counts.idxmax()
min_label = counts.idxmin()
print(f"  Before balance: human={counts.get(0,0):,}  chatgpt={counts.get(1,0):,}")

if counts.max() != counts.min():
    majority_df  = flat_df[flat_df["label"] == maj_label].sample(
        n=min_count, random_state=RANDOM_SEED
    )
    minority_df  = flat_df[flat_df["label"] == min_label]
    balanced_df  = pd.concat([majority_df, minority_df]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    print(f"  Downsampled label {maj_label} from {counts.get(maj_label,0):,} to {min_count:,}")
else:
    balanced_df = flat_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print("  Classes already balanced — no downsampling needed.")

final_counts = balanced_df["label"].value_counts()
print(f"  After balance : human={final_counts.get(0,0):,}  chatgpt={final_counts.get(1,0):,}")
print(f"  Total rows    : {len(balanced_df):,}")


# ── 5. Train / Val / Test split ──────────────────────────────────────────────
print(section("5. Splitting 70 / 15 / 15 (stratified by label)"))

# Split off test (15%)
train_val, test = train_test_split(
    balanced_df, test_size=0.15, stratify=balanced_df["label"],
    random_state=RANDOM_SEED
)
# From remaining 85%, take val as 15/85 ≈ 17.65% to give overall 15%
val_ratio = 0.15 / 0.85
train, val = train_test_split(
    train_val, test_size=val_ratio, stratify=train_val["label"],
    random_state=RANDOM_SEED
)

print(f"  Train : {len(train):,} rows")
print(f"  Val   : {len(val):,} rows")
print(f"  Test  : {len(test):,} rows")


# ── 6. Save CSVs ─────────────────────────────────────────────────────────────
print(section("6. Saving processed splits"))

train.to_csv(PROCESSED / "train.csv", index=False, encoding="utf-8")
val.to_csv  (PROCESSED / "val.csv",   index=False, encoding="utf-8")
test.to_csv (PROCESSED / "test.csv",  index=False, encoding="utf-8")

print(f"  Saved --> {PROCESSED / 'train.csv'}")
print(f"  Saved --> {PROCESSED / 'val.csv'}")
print(f"  Saved --> {PROCESSED / 'test.csv'}")


# ── 7. Final summary ─────────────────────────────────────────────────────────
print(section("FINAL SUMMARY"))
print()
print("  Split     Rows      Human    ChatGPT    Avg Text Len")
print("  " + "-" * 60)
for name, split in [("train", train), ("val", val), ("test", test)]:
    print(split_summary(split, name))

print()
src_counts = balanced_df.groupby("source")["label"].value_counts().unstack(fill_value=0)
print("  Class distribution per source domain:")
print(src_counts.to_string(header=True))
print()
print("  Preprocessing complete!")
