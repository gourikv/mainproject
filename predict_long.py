# -*- coding: utf-8 -*-
"""
predict_long.py
---------------
Wrapper around predict.predict() that handles long texts by splitting them
into ~150-word chunks at sentence boundaries, running inference on each chunk,
and returning a word-count-weighted average of the AI probability.

Does NOT modify predict.py in any way.
"""

import re
from typing import Tuple, Dict, List

from predict import predict, nlp, extract_features, FEATURE_COLS


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex (period / ! / ? followed by space or end)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s.strip()]


def _chunk_sentences(sentences: List[str], target_words: int = 150) -> List[str]:
    """Group sentences into chunks of approximately *target_words* words."""
    chunks: List[str] = []
    current: List[str] = []
    current_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        # If adding this sentence would exceed the target and we already have
        # content, flush the current chunk first.
        if current and current_wc + sent_wc > target_words:
            chunks.append(" ".join(current))
            current = []
            current_wc = 0
        current.append(sent)
        current_wc += sent_wc

    if current:
        chunks.append(" ".join(current))

    return chunks


def _extract_raw_features(text: str) -> dict:
    """Extract raw (unscaled) stylometric features for the full text."""
    doc = nlp(text)
    return extract_features(text, doc)


def predict_long(text: str) -> Tuple[str, Dict[str, float], List[float], dict]:
    """
    Run prediction on *text*, chunking it if it exceeds 150 words.

    Args:
        text: Raw text string.

    Returns:
        (label, probabilities, chunk_ai_probs, raw_features)
        - label: 'Human-written' or 'AI-generated'
        - probabilities: {'Human-written': float, 'AI-generated': float}
        - chunk_ai_probs: list of per-chunk AI probabilities (length 1 when
          the text is short enough to be processed in a single pass).
        - raw_features: dict of the 16 unscaled stylometric features
          computed on the full input text (for explainability).
    """
    # Always extract features on the full text for explanation
    raw_features = _extract_raw_features(text)

    word_count = len(text.strip().split())

    # Short text — single-pass, no chunking needed
    if word_count < 150:
        label, probs = predict(text)
        probs_with_raw = dict(probs)
        probs_with_raw["raw_features"] = raw_features
        return label, probs_with_raw, [probs["AI-generated"]], raw_features

    # Long text — split into ~150-word chunks at sentence boundaries
    sentences = _split_sentences(text)
    chunks = _chunk_sentences(sentences, target_words=150)

    chunk_ai_probs: List[float] = []
    chunk_weights: List[int] = []

    for chunk in chunks:
        _label, probs = predict(chunk)
        wc = len(chunk.split())
        chunk_ai_probs.append(probs["AI-generated"])
        chunk_weights.append(wc)

    # Weighted average of AI probability by word count
    total_weight = sum(chunk_weights)
    weighted_ai = sum(p * w for p, w in zip(chunk_ai_probs, chunk_weights)) / total_weight
    weighted_human = 1.0 - weighted_ai

    label = "AI-generated" if weighted_ai > 0.5 else "Human-written"
    prob_dict = {
        "Human-written": weighted_human,
        "AI-generated": weighted_ai,
        "raw_features": raw_features,
    }

    return label, prob_dict, chunk_ai_probs, raw_features
