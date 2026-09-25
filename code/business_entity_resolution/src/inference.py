"""
Inference — run the full pipeline on test data and write output TSVs.

Steps:
  1. Load + preprocess test sources.
  2. Run blocking → candidate pairs.
  3. Compute features for all candidate pairs.
  4. Run trained model → probabilities.
  5. Apply threshold → final matches.
  6. Write matching_results.tsv and candidate_pairs.tsv.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_OUT, CANDIDATE_OUT, OUTPUT_DIR, SEP,
    BLOCKING_TOP_K, FEATURE_BATCH_SIZE,
)
from preprocess import preprocess_dataframe
from blocking import generate_candidates
from features import compute_features_for_pairs, build_lookups, FEATURE_NAMES


def load_and_preprocess_test():
    """Load and preprocess test source files."""
    print("Loading test data...")
    s1 = pd.read_csv(TEST_S1, sep=SEP, dtype=str, keep_default_na=False)
    s2 = pd.read_csv(TEST_S2, sep=SEP, dtype=str, keep_default_na=False)
    s3 = pd.read_csv(TEST_S3, sep=SEP, dtype=str, keep_default_na=False)
    print(f"  Test S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    print("Preprocessing test data...")
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    return s1, s2, s3


def run_inference(model, threshold, s1_df=None, s2_df=None, s3_df=None):
    """Run the full inference pipeline on test data.

    Returns (matching_dict, candidate_dict).
    """
    # Load test data if not provided
    if s1_df is None:
        s1_df, s2_df, s3_df = load_and_preprocess_test()

    # Step 1: Blocking
    print("\n=== Blocking (test) ===")
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    # Step 2: Build lookups
    print("\nBuilding lookups...")
    s1_lookup = build_lookups(s1_df)
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    cand_lookup = build_lookups(cand_df)

    # Step 3: Create all pairs and compute features + predict
    print("\n=== Prediction ===")
    matching_dict = {}
    candidate_dict = {}

    # Process S1 entities in chunks to manage memory
    s1_ids = list(s1_df["entity_id"])
    chunk_size = 10000

    for chunk_start in tqdm(range(0, len(s1_ids), chunk_size), desc="  Predicting"):
        chunk_end = min(chunk_start + chunk_size, len(s1_ids))
        chunk_ids = s1_ids[chunk_start:chunk_end]

        # Gather pairs for this chunk
        pairs = []
        for s1_id in chunk_ids:
            cands = candidates.get(s1_id, [])
            candidate_dict[s1_id] = cands
            for cand_id in cands:
                pairs.append((s1_id, cand_id))

        if not pairs:
            for s1_id in chunk_ids:
                matching_dict[s1_id] = []
            continue

        # Compute features
        X = compute_features_for_pairs(pairs, s1_lookup, cand_lookup, show_progress=False)

        # Predict
        probas = model.predict_proba(X)[:, 1]

        # Group predictions by S1 entity
        pair_preds = {}
        for idx, (s1_id, cand_id) in enumerate(pairs):
            if s1_id not in pair_preds:
                pair_preds[s1_id] = []
            if probas[idx] >= threshold:
                pair_preds[s1_id].append(cand_id)

        for s1_id in chunk_ids:
            matching_dict[s1_id] = pair_preds.get(s1_id, [])

    return matching_dict, candidate_dict


def write_output(matching_dict: dict, candidate_dict: dict, s1_ids: list):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # matching_results.tsv
    with open(MATCHING_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matches = matching_dict.get(s1_id, [])
            matched_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    print(f"  Written {MATCHING_OUT}")

    # candidate_pairs.tsv
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = candidate_dict.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
    print(f"  Written {CANDIDATE_OUT}")

    # Stats
    total_matches = sum(len(v) for v in matching_dict.values())
    matched_s1 = sum(1 for v in matching_dict.values() if v)
    total_cands = sum(len(v) for v in candidate_dict.values())
    print(f"  Matching: {total_matches} total matches across {matched_s1}/{len(s1_ids)} S1 entities")
    print(f"  Candidates: {total_cands} total candidates")
