"""
Evaluation utilities — compute F0.5 score (macro-averaged per S1 entity).

Used during training/validation. Not needed at inference time.
"""

# TODO: implement
# - Per-entity precision, recall, F0.5
# - Macro-average across all S1 entities
# - Handle singletons: empty prediction on empty truth = 1.0, any prediction on empty truth = 0.0
