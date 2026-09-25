"""
Evaluation — F0.5 score, macro-averaged per S1 entity.

Handles singletons:
  - empty prediction + empty truth → 1.0
  - non-empty prediction + empty truth → 0.0
  - empty prediction + non-empty truth → 0.0
"""

import pandas as pd
from config import SEP


def f05_score(predicted: set, truth: set) -> float:
    """Per-entity F0.5 score."""
    # Both empty → singleton correctly identified
    if not predicted and not truth:
        return 1.0
    # One empty, other not → complete miss
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = 0.25  # 0.5^2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def f05_macro(predictions: dict, ground_truth: dict) -> float:
    """Macro-averaged F0.5 across all S1 entities."""
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        truth = ground_truth[s1_id]
        scores.append(f05_score(pred, truth))
    return sum(scores) / len(scores) if scores else 0.0


def load_ground_truth(path: str) -> dict:
    """Load ground truth TSV → {s1_id: set(matched_ids)}."""
    df = pd.read_csv(path, sep=SEP, dtype=str, keep_default_na=False)
    gt = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()
        gt[s1_id] = set(matched.split(",")) if matched else set()
    return gt


def predictions_to_dict(preds_df: pd.DataFrame) -> dict:
    """Convert predictions DataFrame → {s1_id: set(matched_ids)}."""
    result = {}
    for _, row in preds_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row.get("matched_entity_ids", "")).strip()
        result[s1_id] = set(matched.split(",")) if matched else set()
    return result


if __name__ == "__main__":
    from config import TRAIN_GT
    gt = load_ground_truth(TRAIN_GT)
    total = len(gt)
    singletons = sum(1 for v in gt.values() if not v)
    matched = total - singletons
    avg_matches = sum(len(v) for v in gt.values()) / matched if matched else 0
    print(f"Ground truth: {total} S1 entities")
    print(f"  Singletons (no match): {singletons} ({100*singletons/total:.1f}%)")
    print(f"  With matches: {matched} ({100*matched/total:.1f}%)")
    print(f"  Avg matches per matched entity: {avg_matches:.2f}")
