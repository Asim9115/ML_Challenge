"""
Matching model — train a classifier on features, predict match/no-match.

Uses training ground truth for labels.
Optimizes threshold for F0.5 on validation set.
"""

# TODO: implement
# - Train/val split using config.VAL_SPLIT
# - Train XGBoost (or similar) binary classifier
# - Tune probability threshold to maximize F0.5
# - Save trained model for inference
# - Print validation F0.5 score
