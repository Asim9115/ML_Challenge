"""
Main — orchestrates the full entity resolution pipeline.

Usage:
    python src/main.py                        # full pipeline: train + test
    python src/main.py --train-only           # train + validate only
    python src/main.py --test-only            # inference only (requires trained model)
    python src/main.py --no-optuna            # skip Optuna, use default params
    python src/main.py --optuna-trials 100    # more Optuna trials
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, SEP, RANDOM_SEED, TRAIN_SAMPLE_N, VAL_SPLIT,
    BLOCKING_TOP_K,
)
from preprocess import preprocess_dataframe
from blocking import generate_candidates, measure_blocking_recall
from features import compute_features_for_pairs, build_lookups, FEATURE_NAMES
from evaluate import load_ground_truth, f05_score, f05_macro
from model import (
    create_training_pairs,
    train_xgb_model,
    train_final_model,
    optuna_optimize,
    find_optimal_threshold,
    save_model,
    load_model,
)
from inference import load_and_preprocess_test, run_inference, write_output


def load_and_preprocess_train(sample_n=None):
    """Load training data and preprocess."""
    print("=== Loading training data ===")
    s1 = pd.read_csv(TRAIN_S1, sep=SEP, dtype=str, keep_default_na=False)
    s2 = pd.read_csv(TRAIN_S2, sep=SEP, dtype=str, keep_default_na=False)
    s3 = pd.read_csv(TRAIN_S3, sep=SEP, dtype=str, keep_default_na=False)
    print(f"  Train S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    # Sample S1 entities for tractable training
    if sample_n and sample_n < len(s1):
        s1 = s1.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  Sampled {sample_n} S1 entities for training")

    print("Preprocessing...")
    t0 = time.time()
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    print(f"  Preprocessing done in {time.time()-t0:.1f}s")
    return s1, s2, s3


def train_pipeline(sample_n=None, use_optuna=True, optuna_trials=50):
    """Full training pipeline: load → preprocess → block → featurize → Optuna → train → validate."""
    total_start = time.time()

    # Load data
    s1, s2, s3 = load_and_preprocess_train(sample_n=sample_n)
    gt = load_ground_truth(TRAIN_GT)

    # Filter ground truth to sampled S1 entities
    s1_ids = set(s1["entity_id"])
    gt_filtered = {k: v for k, v in gt.items() if k in s1_ids}
    print(f"  Ground truth entries for sampled S1: {len(gt_filtered)}")

    # Train/val split on S1 entity IDs
    all_s1_ids = sorted(gt_filtered.keys())
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_s1_ids)
    split_idx = int(len(all_s1_ids) * (1 - VAL_SPLIT))
    train_s1_ids = set(all_s1_ids[:split_idx])
    val_s1_ids = set(all_s1_ids[split_idx:])
    print(f"  Train S1: {len(train_s1_ids)}, Val S1: {len(val_s1_ids)}")

    # Split DataFrames
    s1_train = s1[s1["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_s1_ids)].reset_index(drop=True)
    gt_train = {k: v for k, v in gt_filtered.items() if k in train_s1_ids}
    gt_val = {k: v for k, v in gt_filtered.items() if k in val_s1_ids}

    # ── Blocking (train) ──
    print("\n=== Blocking (train split) ===")
    t0 = time.time()
    train_candidates = generate_candidates(s1_train, s2, s3)
    print(f"  Blocking done in {time.time()-t0:.1f}s")
    measure_blocking_recall(train_candidates, gt_train)

    # ── Blocking (val) ──
    print("\n=== Blocking (val split) ===")
    t0 = time.time()
    val_candidates = generate_candidates(s1_val, s2, s3)
    print(f"  Blocking done in {time.time()-t0:.1f}s")
    measure_blocking_recall(val_candidates, gt_val)

    # ── Create training pairs ──
    print("\n=== Creating training pairs ===")
    train_pairs = create_training_pairs(train_candidates, gt_train)
    val_pairs = create_training_pairs(val_candidates, gt_val, neg_pos_ratio=10)

    # ── Build lookups ──
    print("\nBuilding entity lookups...")
    s1_lookup = build_lookups(s1)
    cand_df = pd.concat([s2, s3], ignore_index=True)
    cand_lookup = build_lookups(cand_df)
    del cand_df  # free memory

    # ── Compute features ──
    print("\n=== Computing features (train) ===")
    t0 = time.time()
    pair_list_train = [(p[0], p[1]) for p in train_pairs]
    X_train = compute_features_for_pairs(pair_list_train, s1_lookup, cand_lookup)
    y_train = np.array([p[2] for p in train_pairs], dtype=np.float32)
    print(f"  Train features: {X_train.shape}, done in {time.time()-t0:.1f}s")

    print("\n=== Computing features (val) ===")
    t0 = time.time()
    pair_list_val = [(p[0], p[1]) for p in val_pairs]
    X_val = compute_features_for_pairs(pair_list_val, s1_lookup, cand_lookup)
    y_val = np.array([p[2] for p in val_pairs], dtype=np.float32)
    print(f"  Val features: {X_val.shape}, done in {time.time()-t0:.1f}s")

    # ── Optuna or default training ──
    if use_optuna:
        print("\n=== Optuna Hyperparameter Optimization ===")
        t0 = time.time()
        # Prepare val_pairs with features for threshold search inside Optuna
        val_pairs_with_info = [
            (val_pairs[i][0], val_pairs[i][1], val_pairs[i][2], None)
            for i in range(len(val_pairs))
        ]
        best_params, threshold, best_f05 = optuna_optimize(
            X_train, y_train, X_val, y_val,
            val_pairs, gt_val,
            n_trials=optuna_trials,
        )
        print(f"  Optuna done in {time.time()-t0:.1f}s")

        # Retrain final model with best params on full train data
        print("\n=== Training final model with best params ===")
        t0 = time.time()
        model = train_final_model(X_train, y_train, X_val, y_val, best_params)
        print(f"  Final training done in {time.time()-t0:.1f}s")
    else:
        print("\n=== Training XGBoost (default params) ===")
        t0 = time.time()
        model = train_xgb_model(X_train, y_train, X_val, y_val)
        print(f"  Training done in {time.time()-t0:.1f}s")

        # Find optimal threshold
        print("\n=== Threshold optimization ===")
        threshold = find_optimal_threshold(model, val_pairs, gt_val, s1_lookup, cand_lookup)
        best_params = None

    # ── Feature importance ──
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\n  Feature importance:")
    for name, imp in ranked:
        print(f"    {name:25s} {imp:.4f}")

    # ── Final validation score ──
    print("\n=== Final validation results ===")
    probas_val = model.predict_proba(X_val)[:, 1]
    val_preds = {}
    for idx, (s1_id, cand_id) in enumerate(pair_list_val):
        if probas_val[idx] >= threshold:
            if s1_id not in val_preds:
                val_preds[s1_id] = set()
            val_preds[s1_id].add(cand_id)
    for s1_id in gt_val:
        if s1_id not in val_preds:
            val_preds[s1_id] = set()

    val_f05 = f05_macro(val_preds, gt_val)
    print(f"  Validation F0.5 (macro): {val_f05:.4f}")

    # Save model
    save_model(model, threshold, best_params=best_params)

    total_time = time.time() - total_start
    print(f"\n=== Training complete in {total_time:.1f}s ===")
    return model, threshold


def test_pipeline(model=None, threshold=None):
    """Run inference on test set and write outputs."""
    if model is None:
        model, threshold = load_model()

    print("\n=== Test Inference ===")
    t0 = time.time()
    s1_test, s2_test, s3_test = load_and_preprocess_test()
    matching_dict, candidate_dict = run_inference(model, threshold, s1_test, s2_test, s3_test)

    s1_ids = s1_test["entity_id"].tolist()
    write_output(matching_dict, candidate_dict, s1_ids)
    print(f"\n  Test inference done in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--train-only", action="store_true", help="Train + validate only")
    parser.add_argument("--test-only", action="store_true", help="Inference only")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna, use default params")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--sample-n", type=int, default=TRAIN_SAMPLE_N,
                        help=f"S1 entities to sample for training (default: {TRAIN_SAMPLE_N})")
    args = parser.parse_args()

    use_optuna = not args.no_optuna

    if args.test_only:
        test_pipeline()
    elif args.train_only:
        train_pipeline(sample_n=args.sample_n, use_optuna=use_optuna,
                       optuna_trials=args.optuna_trials)
    else:
        model, threshold = train_pipeline(
            sample_n=args.sample_n, use_optuna=use_optuna,
            optuna_trials=args.optuna_trials,
        )
        test_pipeline(model, threshold)


if __name__ == "__main__":
    main()
