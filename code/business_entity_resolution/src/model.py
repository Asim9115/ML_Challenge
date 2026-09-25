"""
Model — XGBoost binary classifier with Optuna hyperparameter optimization.

Training workflow:
  1. Take blocked candidate pairs from training data.
  2. Label using ground truth (match=1, non-match=0).
  3. Sample negatives to control ratio.
  4. Compute features.
  5. Optuna search over XGBoost params + threshold, optimizing F0.5 directly.
  6. Retrain final model with best params on full train data.
"""

import os
import pickle
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from tqdm import tqdm

from config import (
    NEG_POS_RATIO,
    RANDOM_SEED,
    MODEL_PATH,
    MODEL_DIR,
)
from features import FEATURE_NAMES, compute_features_for_pairs, build_lookups
from evaluate import f05_score


# Suppress Optuna info logs (only show trial results)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_training_pairs(
    candidates: dict,
    ground_truth: dict,
    neg_pos_ratio: int = None,
    seed: int = None,
) -> list:
    """Create labeled (s1_id, cand_id, label) from blocked candidates + ground truth.

    Positive: candidate is in ground truth matches.
    Negative: candidate is NOT in ground truth matches (hard negatives from blocking).
    Negatives are subsampled to neg_pos_ratio * number of positives.
    """
    neg_pos_ratio = neg_pos_ratio or NEG_POS_RATIO
    seed = seed or RANDOM_SEED
    rng = np.random.RandomState(seed)

    positives = []
    negatives = []

    for s1_id, cand_list in candidates.items():
        true_matches = ground_truth.get(s1_id, set())
        for cand_id in cand_list:
            if cand_id in true_matches:
                positives.append((s1_id, cand_id, 1))
            else:
                negatives.append((s1_id, cand_id, 0))

    # Subsample negatives
    max_neg = len(positives) * neg_pos_ratio
    if len(negatives) > max_neg:
        idx = rng.choice(len(negatives), size=max_neg, replace=False)
        negatives = [negatives[i] for i in idx]

    pairs = positives + negatives
    rng.shuffle(pairs)
    print(f"  Training pairs: {len(positives)} positive, {len(negatives)} negative "
          f"(ratio 1:{len(negatives) // max(len(positives), 1)})")
    return pairs


def _compute_val_f05(model, threshold, val_pairs, val_gt):
    """Compute macro F0.5 on validation set given model + threshold."""
    pair_list = [(p[0], p[1]) for p in val_pairs]
    X_val = np.array([p[3] for p in val_pairs])  # pre-computed features stored at index 3
    probas = model.predict_proba(X_val)[:, 1]

    # Group predictions by S1 entity
    predictions = {}
    for idx, (s1_id, cand_id, _, _) in enumerate(val_pairs):
        if probas[idx] >= threshold:
            if s1_id not in predictions:
                predictions[s1_id] = set()
            predictions[s1_id].add(cand_id)

    # Include singletons
    for s1_id in val_gt:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    # Macro F0.5
    scores = []
    for s1_id in val_gt:
        pred = predictions.get(s1_id, set())
        truth = val_gt[s1_id]
        scores.append(f05_score(pred, truth))
    return sum(scores) / len(scores) if scores else 0.0


def optuna_optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pairs: list,
    val_gt: dict,
    n_trials: int = 50,
) -> tuple:
    """Bayesian hyperparameter search using Optuna, optimizing F0.5 directly.

    Searches over:
      - XGBoost: max_depth, learning_rate, n_estimators, subsample,
        colsample_bytree, min_child_weight, scale_pos_weight, gamma, reg_alpha, reg_lambda
      - Threshold: probability cutoff for match decision

    Returns (best_params, best_threshold, best_f05).
    """

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbosity": 0,
            # Searched params
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        n_estimators = params.pop("n_estimators")
        model = XGBClassifier(n_estimators=n_estimators, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Search threshold
        probas = model.predict_proba(X_val)[:, 1]

        # Group by S1 entity
        s1_to_preds = {}
        for idx in range(len(val_pairs)):
            s1_id, cand_id = val_pairs[idx][0], val_pairs[idx][1]
            if s1_id not in s1_to_preds:
                s1_to_preds[s1_id] = []
            s1_to_preds[s1_id].append((cand_id, probas[idx]))

        best_f05 = 0.0
        best_thr = 0.5
        for thr in np.arange(0.15, 0.85, 0.05):
            predictions = {}
            for s1_id, cand_scores in s1_to_preds.items():
                matched = {cid for cid, score in cand_scores if score >= thr}
                if matched:
                    predictions[s1_id] = matched

            for s1_id in val_gt:
                if s1_id not in predictions:
                    predictions[s1_id] = set()

            scores = []
            for s1_id in val_gt:
                pred = predictions.get(s1_id, set())
                truth = val_gt[s1_id]
                scores.append(f05_score(pred, truth))
            macro_f05 = sum(scores) / len(scores) if scores else 0.0

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_thr = thr

        trial.set_user_attr("threshold", best_thr)
        return best_f05

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    print(f"  Running Optuna: {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_threshold = best_trial.user_attrs["threshold"]
    best_f05 = best_trial.value

    print(f"\n  Best trial #{best_trial.number}:")
    print(f"    F0.5 = {best_f05:.4f}, threshold = {best_threshold:.2f}")
    print(f"    Params: {best_params}")

    return best_params, best_threshold, best_f05


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train the final model using best Optuna params on full data."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 1,
    }
    params.update(best_params)
    n_estimators = params.pop("n_estimators")

    model = XGBClassifier(n_estimators=n_estimators, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def train_xgb_model(X_train, y_train, X_val, y_val):
    """Fallback: train with default params (no Optuna). Used if you want a quick run."""
    from config import XGB_PARAMS
    params = dict(XGB_PARAMS)
    n_estimators = params.pop("n_estimators", 500)

    model = XGBClassifier(n_estimators=n_estimators, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def find_optimal_threshold(
    model,
    val_pairs: list,
    val_ground_truth: dict,
    s1_lookup: dict,
    cand_lookup: dict,
) -> float:
    """Find probability threshold that maximizes F0.5 on validation data.
    Used as fallback when not using Optuna (Optuna does joint search).
    """
    # Compute features & predict
    X_val = compute_features_for_pairs(val_pairs, s1_lookup, cand_lookup, show_progress=False)
    probas = model.predict_proba(X_val)[:, 1]

    # Group by S1 entity
    s1_to_preds = {}
    for idx, (s1_id, cand_id, _) in enumerate(val_pairs):
        if s1_id not in s1_to_preds:
            s1_to_preds[s1_id] = []
        s1_to_preds[s1_id].append((cand_id, probas[idx]))

    best_threshold = 0.5
    best_f05 = 0.0

    for threshold in np.arange(0.15, 0.90, 0.05):
        predictions = {}
        for s1_id, cand_scores in s1_to_preds.items():
            matched = {cid for cid, score in cand_scores if score >= threshold}
            predictions[s1_id] = matched

        for s1_id in val_ground_truth:
            if s1_id not in predictions:
                predictions[s1_id] = set()

        scores = []
        for s1_id in val_ground_truth:
            if s1_id in predictions or s1_id in s1_to_preds:
                pred = predictions.get(s1_id, set())
                truth = val_ground_truth[s1_id]
                scores.append(f05_score(pred, truth))
        macro_f05 = sum(scores) / len(scores) if scores else 0.0

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = threshold

    print(f"  Optimal threshold: {best_threshold:.2f} (F0.5 = {best_f05:.4f})")
    return best_threshold


def save_model(model, threshold, best_params=None, path=None):
    """Save model + threshold + params."""
    path = path or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    meta_path = path.replace(".json", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "threshold": threshold,
            "features": FEATURE_NAMES,
            "best_params": best_params,
        }, f)
    print(f"  Model saved to {path}")


def load_model(path=None):
    """Load model + threshold."""
    path = path or MODEL_PATH
    model = XGBClassifier()
    model.load_model(path)
    meta_path = path.replace(".json", "_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    print(f"  Model loaded from {path}, threshold={meta['threshold']:.2f}")
    return model, meta["threshold"]
