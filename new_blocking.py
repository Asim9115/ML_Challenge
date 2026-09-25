"""
new_blocking.py — Generate candidate pairs using TF-IDF cosine similarity.

Run after new_preprocess.py:
    python new_blocking.py --mode train    # blocking on train data
    python new_blocking.py --mode test     # blocking on test data
    python new_blocking.py --mode both     # both

What it does:
  1. Loads preprocessed S1, S2, S3 files
  2. Splits by country
  3. Builds TF-IDF (char n-grams) on S2+S3 combined_text
  4. For each S1 entity, finds top-K most similar S2+S3 candidates
  5. Saves candidates to disk as a TSV

Output:
  dataset/preprocessed/train_candidates.tsv
  dataset/preprocessed/test_candidates.tsv
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm


# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
PREP_DIR = os.path.join(DATA_DIR, "preprocessed")

# ─── Config ───────────────────────────────────────────────
TOP_K = 50                     # candidates per S1 entity
BATCH_SIZE = 2000              # S1 entities per batch
TFIDF_MAX_FEATURES = 80000     # vocabulary cap
TFIDF_NGRAM_RANGE = (2, 4)     # char n-gram range


def load_preprocessed(mode):
    """Load preprocessed TSVs for train or test."""
    prefix = "train" if mode == "train" else "test"
    s1_path = os.path.join(PREP_DIR, f"{prefix}_source1_clean.tsv")
    s2_path = os.path.join(PREP_DIR, f"{prefix}_source2_clean.tsv")
    s3_path = os.path.join(PREP_DIR, f"{prefix}_source3_clean.tsv")

    print(f"Loading {prefix} preprocessed data...")
    t0 = time.time()
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,} — loaded in {time.time()-t0:.1f}s")
    return s1, s2, s3


def top_k_from_sparse_row(sparse_row, k):
    """Get top-k indices and scores from a single sparse matrix row."""
    data = sparse_row.data
    indices = sparse_row.indices
    if len(data) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if len(data) <= k:
        order = np.argsort(-data)
        return indices[order], data[order]
    top_k_pos = np.argpartition(data, -k)[-k:]
    order = np.argsort(-data[top_k_pos])
    top_k_pos = top_k_pos[order]
    return indices[top_k_pos], data[top_k_pos]


def block_one_country(s1_country, cand_country, country_name, top_k=TOP_K):
    """Run TF-IDF blocking for one country partition.

    Returns dict: {s1_entity_id: [candidate_entity_ids]}
    """
    cand_texts = cand_country["combined_text"].tolist()
    cand_ids = cand_country["entity_id"].tolist()
    s1_texts = s1_country["combined_text"].tolist()
    s1_ids = s1_country["entity_id"].tolist()

    if len(cand_texts) == 0:
        print(f"    No candidates for {country_name} — all singletons")
        return {eid: [] for eid in s1_ids}

    # Build TF-IDF on candidates
    print(f"    Building TF-IDF on {len(cand_texts):,} candidates...", end=" ", flush=True)
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    cand_matrix = vectorizer.fit_transform(cand_texts)
    cand_matrix = normalize(cand_matrix, norm="l2")
    print(f"done in {time.time()-t0:.1f}s (vocab: {len(vectorizer.vocabulary_):,})")

    # Query S1 in batches
    results = {}
    n_batches = (len(s1_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in tqdm(range(0, len(s1_texts), BATCH_SIZE),
                            total=n_batches, desc=f"    Querying [{country_name}]"):
        batch_end = min(batch_start + BATCH_SIZE, len(s1_texts))
        batch_texts = s1_texts[batch_start:batch_end]
        batch_ids = s1_ids[batch_start:batch_end]

        # Transform and normalize query batch
        query_matrix = vectorizer.transform(batch_texts)
        query_matrix = normalize(query_matrix, norm="l2")

        # Sparse dot product = cosine similarity (both L2-normalized)
        similarity = query_matrix.dot(cand_matrix.T)

        # Extract top-K per query
        for i in range(similarity.shape[0]):
            row = similarity.getrow(i)
            indices, scores = top_k_from_sparse_row(row, top_k)
            # Only keep candidates with score > 0
            mask = scores > 0
            candidate_list = [cand_ids[idx] for idx in indices[mask]]
            results[batch_ids[i]] = candidate_list

    return results


def run_blocking(mode, top_k=TOP_K):
    """Run full blocking pipeline for train or test."""
    print(f"\n{'='*60}")
    print(f"BLOCKING — {mode.upper()}")
    print(f"{'='*60}")

    total_start = time.time()
    s1, s2, s3 = load_preprocessed(mode)

    # Combine S2 + S3 as candidate pool
    candidates = pd.concat([s2, s3], ignore_index=True)
    del s2, s3  # free memory

    # Get all countries
    countries = sorted(s1["country"].unique())
    print(f"\nCountries found: {countries}")

    all_candidates = {}

    for country in countries:
        s1_c = s1[s1["country"] == country].reset_index(drop=True)
        cand_c = candidates[candidates["country"] == country].reset_index(drop=True)

        print(f"\n  [{country}] S1: {len(s1_c):,}, Candidates: {len(cand_c):,}")
        t0 = time.time()

        country_results = block_one_country(s1_c, cand_c, country, top_k=top_k)
        all_candidates.update(country_results)

        print(f"    Done in {time.time()-t0:.1f}s")

    # Make sure every S1 entity has an entry
    for eid in s1["entity_id"]:
        if eid not in all_candidates:
            all_candidates[eid] = []

    # Stats
    total_pairs = sum(len(v) for v in all_candidates.values())
    non_empty = sum(1 for v in all_candidates.values() if v)
    avg_cands = total_pairs / len(all_candidates) if all_candidates else 0
    print(f"\n  Total candidate pairs: {total_pairs:,}")
    print(f"  S1 with candidates: {non_empty:,} / {len(all_candidates):,}")
    print(f"  Avg candidates per S1: {avg_cands:.1f}")

    # Save to disk
    out_path = os.path.join(PREP_DIR, f"{mode}_candidates.tsv")
    print(f"\n  Saving to {out_path}...")
    t0 = time.time()

    # Use S1 order from original dataframe
    s1_ids_ordered = s1["entity_id"].tolist()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_ordered:
            cands = all_candidates.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved in {time.time()-t0:.1f}s ({size_mb:.1f} MB)")

    # If train mode, check blocking recall against ground truth
    if mode == "train":
        gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            check_blocking_recall(all_candidates, gt_path)

    total_time = time.time() - total_start
    print(f"\n  Blocking ({mode}) complete in {total_time:.1f}s")


def check_blocking_recall(candidates, gt_path):
    """Check what fraction of true matches survived blocking."""
    print(f"\n  Checking blocking recall against ground truth...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    total_true = 0
    found_true = 0
    missed_examples = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()
        if not matched:
            continue
        true_matches = set(matched.split(","))
        total_true += len(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        found = len(true_matches & cand_set)
        found_true += found
        if found < len(true_matches) and len(missed_examples) < 5:
            missed = true_matches - cand_set
            missed_examples.append((s1_id, missed))

    recall = found_true / total_true if total_true > 0 else 1.0
    print(f"  Blocking recall: {found_true:,} / {total_true:,} = {recall:.4f}")
    print(f"  Missed matches: {total_true - found_true:,}")

    if missed_examples:
        print(f"\n  Sample missed matches:")
        for s1_id, missed in missed_examples:
            print(f"    {s1_id} missed: {missed}")


def main():
    parser = argparse.ArgumentParser(description="TF-IDF Blocking for Entity Resolution")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="train",
                        help="Which data to block (default: train)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Candidates per S1 entity (default: {TOP_K})")
    args = parser.parse_args()

    top_k = args.top_k

    if args.mode == "both":
        run_blocking("train", top_k)
        run_blocking("test", top_k)
    else:
        run_blocking(args.mode, top_k)


if __name__ == "__main__":
    main()
