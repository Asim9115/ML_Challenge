"""
Feature engineering — build similarity features for candidate pairs.

Features computed per (S1, candidate) pair:
  Name:  rapidfuzz ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler, Jaccard
  Addr:  rapidfuzz ratio, token_sort_ratio, token overlap, numeric token match
  Other: country match, name/addr length ratios
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

from config import FEATURE_BATCH_SIZE
from preprocess import extract_numeric_tokens


# ── Feature names (order matters for DataFrame columns) ──
FEATURE_NAMES = [
    # Name similarity
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_jaccard_tokens",
    "name_jaccard_bigrams",
    "name_len_ratio",
    "name_exact",
    # Address similarity
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard_tokens",
    "addr_numeric_overlap",
    "addr_len_ratio",
    "addr_both_empty",
    # Cross-field
    "country_match",
    "combined_ratio",
]


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _char_bigrams(text: str) -> set:
    """Character bigrams of a string."""
    if len(text) < 2:
        return set()
    return {text[i:i+2] for i in range(len(text) - 1)}


def _safe_len_ratio(a: str, b: str) -> float:
    """Length ratio min/max, 0 if both empty."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _numeric_overlap(a: str, b: str) -> float:
    """Fraction of numeric tokens shared between two strings."""
    nums_a = extract_numeric_tokens(a)
    nums_b = extract_numeric_tokens(b)
    if not nums_a and not nums_b:
        return 1.0
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a | nums_b)


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_country: str,
    cand_name: str, cand_addr: str, cand_country: str,
) -> list:
    """Compute feature vector for a single (S1, candidate) pair.

    Returns a list of floats in FEATURE_NAMES order.
    """
    # ── Name features ──
    name_ratio = rfuzz.ratio(s1_name, cand_name) / 100.0
    name_partial = rfuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_token_sort = rfuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_token_set = rfuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_jw = JaroWinkler.similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_jaccard_tok = _jaccard(set(s1_name.split()), set(cand_name.split()))
    name_jaccard_bi = _jaccard(_char_bigrams(s1_name), _char_bigrams(cand_name))
    name_len_ratio = _safe_len_ratio(s1_name, cand_name)
    name_exact = 1.0 if s1_name == cand_name and s1_name else 0.0

    # ── Address features ──
    addr_ratio = rfuzz.ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
    addr_token_sort = rfuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
    addr_token_set = rfuzz.token_set_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
    addr_jaccard_tok = _jaccard(set(s1_addr.split()), set(cand_addr.split())) if s1_addr and cand_addr else 0.0
    addr_numeric = _numeric_overlap(s1_addr, cand_addr)
    addr_len_ratio = _safe_len_ratio(s1_addr, cand_addr)
    addr_both_empty = 1.0 if (not s1_addr and not cand_addr) else 0.0

    # ── Cross-field ──
    country_match = 1.0 if s1_country == cand_country else 0.0
    combined_s1 = (s1_name + " " + s1_addr).strip()
    combined_cand = (cand_name + " " + cand_addr).strip()
    combined_ratio = rfuzz.token_sort_ratio(combined_s1, combined_cand) / 100.0

    return [
        name_ratio, name_partial, name_token_sort, name_token_set,
        name_jw, name_jaccard_tok, name_jaccard_bi, name_len_ratio, name_exact,
        addr_ratio, addr_token_sort, addr_token_set, addr_jaccard_tok,
        addr_numeric, addr_len_ratio, addr_both_empty,
        country_match, combined_ratio,
    ]


def compute_features_for_pairs(
    pairs: list,
    s1_lookup: dict,
    cand_lookup: dict,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute features for a list of (s1_id, cand_id) pairs.

    Args:
        pairs: list of (s1_entity_id, candidate_entity_id) tuples
        s1_lookup: {entity_id: (name_norm, addr_norm, country)}
        cand_lookup: {entity_id: (name_norm, addr_norm, country)}

    Returns:
        np.ndarray of shape (len(pairs), len(FEATURE_NAMES))
    """
    features = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)

    iterator = range(len(pairs))
    if show_progress and len(pairs) > 10000:
        iterator = tqdm(iterator, desc="  Computing features")

    for i in iterator:
        s1_id, cand_id = pairs[i]
        s1_name, s1_addr, s1_country = s1_lookup.get(s1_id, ("", "", ""))
        c_name, c_addr, c_country = cand_lookup.get(cand_id, ("", "", ""))
        features[i] = compute_pair_features(
            s1_name, s1_addr, s1_country,
            c_name, c_addr, c_country,
        )
    return features


def build_lookups(df: pd.DataFrame) -> dict:
    """Build {entity_id: (name_norm, addr_norm, country)} from a preprocessed DataFrame."""
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["entity_id"]] = (
            row.get("name_norm", ""),
            row.get("addr_norm", ""),
            row.get("country", ""),
        )
    return lookup
