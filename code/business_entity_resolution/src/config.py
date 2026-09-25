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

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ── Constants ──────────────────────────────────────────
SEPARATOR = "\t"

# ── Hyperparams (tune later) ──────────────────────────
BLOCKING_TOP_K = 50          # max candidates per S1 entity from blocking
MATCH_THRESHOLD = 0.5        # classifier probability threshold
RANDOM_SEED = 42
VAL_SPLIT = 0.2              # fraction of train held out for validation
