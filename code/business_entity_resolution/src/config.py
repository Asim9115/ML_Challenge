"""
Shared config — paths, constants, hyperparams.
Everything else imports from here.
"""

import os

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "models")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_matcher.json")

# ── Constants ──────────────────────────────────────────
SEP = "\t"

# ── Hyperparams ───────────────────────────────────────
BLOCKING_TOP_K = 50            # max candidates per S1 entity from blocking
MATCH_THRESHOLD = 0.5          # classifier probability threshold (tuned later)
RANDOM_SEED = 42
VAL_SPLIT = 0.2                # fraction of S1 entities held out for validation
TRAIN_SAMPLE_N = 80000         # number of S1 entities to sample for training (None = all)
NEG_POS_RATIO = 5              # negative:positive ratio for training pairs
TFIDF_MAX_FEATURES = 80000     # TF-IDF vocabulary cap
TFIDF_NGRAM_RANGE = (2, 4)     # char n-gram range for TF-IDF
BLOCKING_BATCH_SIZE = 2000     # S1 entities processed per blocking batch
FEATURE_BATCH_SIZE = 50000     # pairs processed per feature batch
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 7,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 3,       # bias toward precision
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}
