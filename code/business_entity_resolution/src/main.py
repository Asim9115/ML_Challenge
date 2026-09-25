"""
Main — orchestrates the full entity resolution pipeline.

Usage:
    python src/main.py

Steps:
    1. Load & preprocess data
    2. Blocking (candidate generation)
    3. Feature engineering
    4. Train matching model (on train split)
    5. Evaluate on validation split
    6. Run inference on test set
    7. Write output TSVs
"""

# TODO: implement
# Wire up all modules: config → preprocess → blocking → features → model → inference
