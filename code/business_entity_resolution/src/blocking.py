"""
Blocking — candidate generation via TF-IDF cosine similarity.

Strategy:
  1. Partition by country (only compare within same country).
  2. Build TF-IDF (char n-grams) on combined name+address for S2+S3.
  3. For each batch of S1, compute sparse cosine dot product → top-K candidates.
  4. Return {s1_id: [candidate_ids]}.

This sets the recall ceiling — any true match not in candidates is lost forever.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import (
    BLOCKING_TOP_K,
    BLOCKING_BATCH_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)


def _top_k_from_sparse_row(sparse_row, k):
    """Get top-k indices and scores from a single sparse row."""
    data = sparse_row.data
    indices = sparse_row.indices
    if len(data) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if len(data) <= k:
        order = np.argsort(-data)
        return indices[order], data[order]
    # Partial sort — faster than full sort for large arrays
    top_k_pos = np.argpartition(data, -k)[-k:]
    order = np.argsort(-data[top_k_pos])
    top_k_pos = top_k_pos[order]
    return indices[top_k_pos], data[top_k_pos]


def build_blocker_for_partition(candidates_texts, max_features=None, ngram_range=None):
    """Fit TF-IDF vectorizer on candidate texts. Return (vectorizer, tfidf_matrix)."""
    max_features = max_features or TFIDF_MAX_FEATURES
    ngram_range = ngram_range or TFIDF_NGRAM_RANGE

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(candidates_texts)
    # L2-normalize rows (should already be by default, but ensure)
    from sklearn.preprocessing import normalize
    tfidf_matrix = normalize(tfidf_matrix, norm="l2")
    return vectorizer, tfidf_matrix


def query_blocker_batch(vectorizer, cand_matrix, query_texts, top_k):
    """Query the blocker for a batch of S1 entities.

    Returns list of (top_k_indices, top_k_scores) per query.
    """
    from sklearn.preprocessing import normalize
    query_matrix = vectorizer.transform(query_texts)
    query_matrix = normalize(query_matrix, norm="l2")
    # Sparse dot product → cosine similarity (both are L2-normalized)
    similarity = query_matrix.dot(cand_matrix.T)

    results = []
    # Work row by row from the sparse result
    if sparse.issparse(similarity):
        for i in range(similarity.shape[0]):
            row = similarity.getrow(i)
            indices, scores = _top_k_from_sparse_row(row, top_k)
            results.append((indices, scores))
    else:
        # Dense fallback (shouldn't happen, but safety)
        for i in range(similarity.shape[0]):
            row = similarity[i]
            if len(row) <= top_k:
                order = np.argsort(-row)
                results.append((order, row[order]))
            else:
                top_idx = np.argpartition(row, -top_k)[-top_k:]
                order = np.argsort(-row[top_idx])
                results.append((top_idx[order], row[top_idx[order]]))
    return results


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = None,
    batch_size: int = None,
    show_progress: bool = True,
) -> dict:
    """Generate blocking candidates for each S1 entity.

    Args:
        s1_df: Source 1 DataFrame (must have entity_id, combined_text, country)
        s2_df: Source 2 DataFrame
        s3_df: Source 3 DataFrame
        top_k: max candidates per S1 entity
        batch_size: S1 entities per batch

    Returns:
        {s1_entity_id: list of candidate_entity_ids}
    """
    top_k = top_k or BLOCKING_TOP_K
    batch_size = batch_size or BLOCKING_BATCH_SIZE

    # Combine S2 + S3 as the candidate pool
    candidates = pd.concat([s2_df, s3_df], ignore_index=True)

    # Get all countries present in S1
    countries = sorted(s1_df["country"].unique())
    all_candidates = {}

    for country in countries:
        s1_country = s1_df[s1_df["country"] == country].reset_index(drop=True)
        cand_country = candidates[candidates["country"] == country].reset_index(drop=True)

        if len(s1_country) == 0:
            continue

        if len(cand_country) == 0:
            # No candidates for this country — all singletons
            for eid in s1_country["entity_id"]:
                all_candidates[eid] = []
            print(f"  [{country}] {len(s1_country)} S1, 0 candidates → all singletons")
            continue

        print(f"  [{country}] {len(s1_country)} S1 entities, {len(cand_country)} candidates")

        # Build TF-IDF blocker on candidates
        cand_texts = cand_country["combined_text"].tolist()
        cand_ids = cand_country["entity_id"].tolist()
        vectorizer, cand_matrix = build_blocker_for_partition(cand_texts)

        # Query S1 in batches
        s1_texts = s1_country["combined_text"].tolist()
        s1_ids = s1_country["entity_id"].tolist()

        n_batches = (len(s1_texts) + batch_size - 1) // batch_size
        iterator = range(0, len(s1_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc=f"  Blocking [{country}]")

        for start in iterator:
            end = min(start + batch_size, len(s1_texts))
            batch_texts = s1_texts[start:end]
            batch_ids = s1_ids[start:end]
            batch_results = query_blocker_batch(
                vectorizer, cand_matrix, batch_texts, top_k
            )
            for j, (indices, scores) in enumerate(batch_results):
                eid = batch_ids[j]
                cand_list = [cand_ids[idx] for idx in indices if scores[list(indices).index(idx)] > 0]
                all_candidates[eid] = cand_list

        # Free memory
        del vectorizer, cand_matrix, cand_texts

    # Ensure every S1 entity has an entry (even if empty)
    for eid in s1_df["entity_id"]:
        if eid not in all_candidates:
            all_candidates[eid] = []

    total_pairs = sum(len(v) for v in all_candidates.values())
    non_empty = sum(1 for v in all_candidates.values() if v)
    print(f"  Blocking done: {total_pairs} total candidate pairs, "
          f"{non_empty}/{len(all_candidates)} S1 entities have candidates")
    return all_candidates


def measure_blocking_recall(candidates: dict, ground_truth: dict) -> float:
    """What fraction of true matches survived blocking?"""
    total_true = 0
    found_true = 0
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true += len(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        found_true += len(true_matches & cand_set)
    recall = found_true / total_true if total_true > 0 else 1.0
    print(f"  Blocking recall: {found_true}/{total_true} = {recall:.4f}")
    return recall
