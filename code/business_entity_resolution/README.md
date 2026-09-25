# Business Entity Resolution Pipeline

## Quick Start

```bash
pip install -r requirements.txt
python src/main.py
```

## Pipeline Steps

1. **EDA** — `src/eda.py` — explore the data, understand noise patterns
2. **Preprocessing** — `src/preprocess.py` — clean/normalize names & addresses
3. **Blocking** — `src/blocking.py` — candidate generation to reduce comparison space
4. **Feature Engineering** — `src/features.py` — build similarity features for candidate pairs
5. **Matching Model** — `src/model.py` — train classifier, predict matches
6. **Inference** — `src/inference.py` — run pipeline on test set, produce output TSVs
7. **Main** — `src/main.py` — orchestrates the full pipeline end-to-end

## Output

- `output/matching_results.tsv` — final matches (uploaded to leaderboard)
- `output/candidate_pairs.tsv` — blocking candidate set

## Validation

```bash
cd 6ab10eb3b23ba_student_resource/student_resource
python3 utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test
```
