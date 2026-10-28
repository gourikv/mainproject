# -*- coding: utf-8 -*-
"""
load_data.py
------------
Loads the HC3 (Human ChatGPT Comparison Corpus) dataset from Hugging Face,
prints its structure, shows sample rows, and saves summary statistics to
data/dataset_summary.txt.

Compatibility: works with HuggingFace datasets v2-v5 / huggingface_hub v0.19+
Windows-safe: uses UTF-8 stdout and ASCII-only print symbols.
"""

import sys
import io
import textwrap
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_TXT = DATA_DIR / "dataset_summary.txt"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "Hello-SimpleAI/HC3"


# ── Helper ───────────────────────────────────────────────────────────────────
def section(title: str, width: int = 70) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def truncate(text, max_chars: int = 300) -> str:
    if not text:
        return "<empty>"
    if isinstance(text, list):
        text = " | ".join(str(t) for t in text)
    return textwrap.shorten(str(text), width=max_chars, placeholder=" ...")


# ── 1. Discover files on the Hub ─────────────────────────────────────────────
print(section("1. Discovering HC3 files on Hugging Face Hub"))
all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
print(f"  Total files in repo: {len(all_files)}")

# Filter to data files (json/jsonl/parquet) for the "all" config
data_files = [f for f in all_files if f.endswith((".json", ".jsonl", ".parquet"))]
print(f"  Data files found   : {len(data_files)}")
for f in data_files[:10]:
    print(f"    {f}")
if len(data_files) > 10:
    print(f"    ... and {len(data_files) - 10} more")


# ── 2. Load Dataset ──────────────────────────────────────────────────────────
print(section("2. Loading HC3 Dataset"))

# Try parquet files first; fall back to json
parquet_files = [f for f in data_files if f.endswith(".parquet")]
json_files    = [f for f in data_files if f.endswith((".json", ".jsonl"))]

dataset = None
if parquet_files:
    try:
        hf_parquet = [f"hf://datasets/{REPO_ID}/{f}" for f in parquet_files]
        print(f"  Loading {len(hf_parquet)} parquet file(s) ...")
        dataset = load_dataset("parquet", data_files={"train": hf_parquet})
        print("  [OK] Dataset loaded via parquet!")
    except Exception as e:
        print(f"  [WARN] Parquet load failed: {e}")

if dataset is None and json_files:
    try:
        hf_json = [f"hf://datasets/{REPO_ID}/{f}" for f in json_files]
        print(f"  Loading {len(hf_json)} JSON file(s) ...")
        dataset = load_dataset("json", data_files={"train": hf_json})
        print("  [OK] Dataset loaded via JSON!")
    except Exception as e:
        print(f"  [FAIL] JSON load failed: {e}")

if dataset is None:
    print("  [FAIL] Could not load dataset. Check network / HF token.")
    sys.exit(1)


# ── 3. Dataset Structure ─────────────────────────────────────────────────────
print(section("3. Dataset Structure"))
print(f"  Splits available : {list(dataset.keys())}")
for split_name, split_data in dataset.items():
    print(f"\n  Split : '{split_name}'")
    print(f"    Rows    : {len(split_data):,}")
    print(f"    Columns : {split_data.column_names}")


# ── 4. Convert to DataFrame ──────────────────────────────────────────────────
df = dataset["train"].to_pandas()
print(f"\n  DataFrame shape : {df.shape}")
print(f"  Columns         : {list(df.columns)}")

# Flatten list-typed answer columns to plain strings
for col in ["human_answers", "chatgpt_answers"]:
    if col in df.columns:
        df[col.replace("_answers", "_text")] = df[col].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )

human_col  = "human_text"  if "human_text"  in df.columns else df.columns[0]
chatgpt_col = "chatgpt_text" if "chatgpt_text" in df.columns else df.columns[1]


# ── 5. Sample Rows ───────────────────────────────────────────────────────────
print(section("4. Sample Rows (first 3 records)"))
for i, row in df.head(3).iterrows():
    print(f"\n  -- Record #{i+1} --")
    print(f"  Source   : {row.get('source', 'N/A')}")
    print(f"  Question : {truncate(row.get('question', ''))}")
    print(f"  Human Ans: {truncate(row.get(human_col, ''))}")
    print(f"  GPT Ans  : {truncate(row.get(chatgpt_col, ''))}")


# ── 6. Statistics & Save ─────────────────────────────────────────────────────
print(section("5. Dataset Statistics"))

df["human_len"]   = df[human_col].str.len()
df["chatgpt_len"] = df[chatgpt_col].str.len()

stats_lines = []
stats_lines.append("HC3 Dataset Summary Statistics")
stats_lines.append("=" * 50)
stats_lines.append(f"Total records : {len(df):,}")
stats_lines.append(f"Columns       : {list(df.columns)}")
stats_lines.append("")

if "source" in df.columns:
    print("\n  Row count per domain (source):")
    stats_lines.append("Row count per domain (source):")
    domain_counts = df["source"].value_counts()
    for domain, count in domain_counts.items():
        line = f"    {str(domain):<35} {count:>6,} rows"
        print(line)
        stats_lines.append(line)

stats_lines.append("")
stats_lines.append("Average text length (characters):")
avg_h  = df["human_len"].mean()
avg_c  = df["chatgpt_len"].mean()
stats_lines.append(f"    Human answers  : {avg_h:.1f} chars")
stats_lines.append(f"    ChatGPT answers: {avg_c:.1f} chars")

print(f"\n  Avg human answer length   : {avg_h:.1f} chars")
print(f"  Avg ChatGPT answer length : {avg_c:.1f} chars")

if "source" in df.columns:
    stats_lines.append("")
    stats_lines.append("Average text length per domain:")
    domain_stats = df.groupby("source")[["human_len", "chatgpt_len"]].mean().round(1)
    stats_lines.append(domain_stats.to_string())
    print("\n  Avg text length per domain:")
    print(domain_stats.to_string())

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(stats_lines))

print(f"\n  Summary saved --> {OUTPUT_TXT}")
print(section("DONE - HC3 Loaded and Verified Successfully!"))
print(f"\n  Total rows   : {len(df):,}")
print(f"  Human cols   : {human_col}")
print(f"  ChatGPT cols : {chatgpt_col}")
print(f"  Source domains: {df['source'].nunique() if 'source' in df.columns else 'N/A'}\n")
