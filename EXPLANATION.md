# Entity Resolution Challenge — Full Breakdown

---

## 1. What's the Aim?

You have **3 databases** of businesses. Same business can appear in multiple databases but with **different names, addresses, typos, languages**.

**Your job:** For each business in Source 1, find all its copies in Source 2 and Source 3.

That's it. Match duplicates across databases.

**Example:**
```
Source 1: "B+ Retail Inc" at "1712 Montebello Ave, Phoenix, AZ"
Source 2: "B Plus Retail Incorporated" at "1712 Montebello Avenue, Phoenix, Arizona"
Source 3: "B+ Retail" at "Montebello Ave, Phoenix"
→ These are the SAME business. Your model should link them.
```

---

## 2. The Dataset

All files are **tab-separated (.tsv)** — NOT comma-separated. Read with `sep="\t"`.

### Training Data
| File | What | Rows |
|------|------|------|
| `train_source1.tsv` | Reference businesses (anchor) | ~2.2M |
| `train_source2.tsv` | Noisy copy database 2 | ~5.0M |
| `train_source3.tsv` | Noisy copy database 3 | ~5.3M |
| `train_ground_truth.tsv` | Answer key: which S2/S3 match each S1 | ~2.2M |

### Test Data
| File | What | Rows |
|------|------|------|
| `test_source1.tsv` | Reference businesses | ~1.7M |
| `test_source2.tsv` | Noisy copy database 2 | ~4.9M |
| `test_source3.tsv` | Noisy copy database 3 | ~5.1M |

### Columns in Every Source File
| Column | What | Example |
|--------|------|---------|
| `entity_id` | Unique ID with prefix `S1-`, `S2-`, `S3-` | `S1-925783039` |
| `business_name` | Name of business (noisy) | `Orelee's Barbershop` |
| `business_address` | Address (noisy, sometimes empty) | `1795 Westchester Drive, High Point, NC` |
| `country` | Country label | `US`, `India`, or `France` (test only) |

### Ground Truth Format
| Column | What |
|--------|------|
| `source1_entity_id` | An S1 entity |
| `matched_entity_ids` | Comma-separated S2/S3 IDs that match it. Empty = no matches (singleton) |

### Key Traps
- **France appears ONLY in test data.** Training has only US and India. Your pipeline must handle unseen countries — don't hardcode country lists.
- **Many S1 entities have NO matches** (singletons). Predicting empty correctly = score 1.0 for that entity. Predicting wrong matches on a singleton = score 0.0.
- **Addresses can be empty.** Some records have no address at all.
- **Hindi/Devanagari text** in Source 2 Indian records. Like `राम मार्केटिंग प्राइवेट लिमिटेड`.
- **French accented text** in test. Like `Café René & Fils`.

---

## 3. What Noise to Expect

This is the core challenge. Same business looks different across sources:

### Name Noise
| Type | Example |
|------|---------|
| Abbreviations | `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited` |
| Legal suffix differences | `Inc` present in one, missing in another |
| Punctuation | `&` vs `and`, missing periods, extra hyphens |
| Typos | `Barbershop` vs `Barbershp` |
| Word order | `Smith & Jones LLC` vs `Jones Smith LLC` |
| DBA/trade names | Completely different name for same business |
| Transliteration | Hindi name in Devanagari vs romanized English |

### Address Noise
| Type | Example |
|------|---------|
| Abbreviations | `Rd` vs `Road`, `St` vs `Street`, `Ave` vs `Avenue` |
| Missing components | No ZIP code, no state, no street number |
| Reordering | `Greensboro, NC, 19 Stardust Trail` vs `19 Stardust Trail, Greensboro, NC` |
| Landmarks | `Near SBI ATM, Main Road` (India-specific) |
| Format variations | `PO Box 6009` vs `P.O. Box 6009` |

---

## 4. Preprocessing — What, How, Why

### Why Preprocess?
Without it, `"Corp"` and `"Corporation"` look completely different to any algorithm. Preprocessing makes similar things look similar before comparison.

### What to Do (in order)

**Step 1: Lowercase everything**
```
"B+ Retail Inc" → "b+ retail inc"
```
Why: Case shouldn't matter for matching.

**Step 2: Transliterate Unicode → ASCII**
```
"राम मार्केटिंग" → "ram marketinga" (via unidecode)
"Café René" → "cafe rene"
```
Why: Hindi Devanagari and French accents need to be comparable with English text. The `unidecode` library converts any script to its closest ASCII representation.

**Step 3: Replace `&` with `and`**
```
"Smith & Jones" → "Smith and Jones"
```

**Step 4: Expand abbreviations**
```
Name:    "Corp" → "Corporation", "Pvt" → "Private", "Ltd" → "Limited"
Address: "Rd" → "Road", "St" → "Street", "Ave" → "Avenue", "Apt" → "Apartment"
```
Why: Standardizes variations so string similarity scores are higher for actual matches.

**Step 5: Strip punctuation**
```
"b+ retail, inc." → "b  retail  inc"
```
Why: Punctuation is inconsistent across sources.

**Step 6: Collapse whitespace**
```
"b  retail  inc" → "b retail inc"
```

### Libraries Used
- `unidecode` — script transliteration
- `re` (built-in) — regex for substitutions

---

## 5. The Pipeline — Step by Step

The problem has **millions** of records. You can't compare every S1 entity against every S2+S3 entity (2M × 10M = 20 trillion comparisons). So the pipeline is:

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐
│ Preprocess  │ ──→ │ Blocking │ ──→ │ Features │ ──→ │  Model  │ ──→ │ Output │
│ Clean text  │     │ Narrow   │     │ Compare  │     │ Decide  │     │  TSVs  │
│             │     │ candidates│    │ pairs    │     │ match?  │     │        │
└─────────────┘     └──────────┘     └──────────┘     └─────────┘     └────────┘
```

### Step 1: Preprocess
Clean all text as described above.

### Step 2: Blocking (Candidate Generation)
**What:** For each S1 entity, quickly find a shortlist of ~50 plausible candidates from S2+S3. Throw away the other 10 million.

**How (TF-IDF approach):**
1. Split S1 and S2+S3 by country (only compare within same country).
2. Build a TF-IDF matrix on the combined `name + address` text of all S2+S3 records using character n-grams (e.g., 2-4 char windows).
3. For each S1 entity, compute cosine similarity against all S2+S3 in that country.
4. Keep top-K (e.g., 50) most similar candidates.

**Why TF-IDF char n-grams?**
- Character n-grams handle typos (e.g., `"Barbershop"` and `"Barbershp"` share most 3-grams).
- TF-IDF downweights common terms (like `"Inc"`, `"Road"`) and upweights rare/distinctive terms.
- Cosine similarity is fast with sparse matrices.

**Why country partitioning?**
- A business in India won't match a business in the US. Cuts the comparison space by ~3x.
- But DON'T hardcode countries — use whatever countries appear in the data.

**Why this matters:**
This step sets the **recall ceiling**. If a true match isn't in your candidate set, you can never find it later. So blocking must be generous — favor recall over precision here.

**Alternative blocking strategies:**
| Strategy | Pros | Cons |
|----------|------|------|
| TF-IDF cosine | Handles typos, scalable with sparse matrices | Misses completely different names (DBA) |
| Token inverted index | Very fast lookup | Misses typo variants |
| Phonetic (Soundex/Metaphone) | Good for pronunciation-based matches | Only works for English names |
| Exact name match | Zero false positives | Way too strict, terrible recall |
| Multiple passes (union) | Best recall | Slower, more candidates to score |

**Best practice:** Use multiple blocking strategies and **union** the candidates. More candidates = higher recall ceiling.

### Step 3: Feature Engineering
**What:** For each (S1 entity, candidate) pair, compute similarity scores that describe how similar the name and address are.

**Features to compute:**

#### Name Features
| Feature | What it measures | Library |
|---------|-----------------|---------|
| Levenshtein ratio | Edit distance (normalized 0-1) | `rapidfuzz.fuzz.ratio` |
| Partial ratio | Best substring match | `rapidfuzz.fuzz.partial_ratio` |
| Token sort ratio | Sort words then compare (handles reordering) | `rapidfuzz.fuzz.token_sort_ratio` |
| Token set ratio | Ignore duplicate/extra words | `rapidfuzz.fuzz.token_set_ratio` |
| Jaro-Winkler | Good for short strings, weighted toward prefix | `rapidfuzz.distance.JaroWinkler` |
| Jaccard (tokens) | Word overlap: \|intersection\| / \|union\| | Manual |
| Jaccard (char bigrams) | Character-level overlap | Manual |
| Length ratio | min(len)/max(len) — flags very different lengths | Manual |
| Exact match flag | 1 if names are identical, 0 otherwise | Manual |

#### Address Features
| Feature | What it measures |
|---------|-----------------|
| Levenshtein ratio | Overall address similarity |
| Token sort ratio | Handles component reordering |
| Token set ratio | Handles missing components |
| Jaccard (tokens) | Word overlap |
| Numeric overlap | Do street numbers / ZIP codes match? |
| Length ratio | Flags missing components |
| Both empty flag | Both addresses missing (can't penalize) |

#### Cross Features
| Feature | What it measures |
|---------|-----------------|
| Country match | Same country? (should always be 1 after blocking, but safety) |
| Combined ratio | Token sort ratio on `name + address` together |

**Total: ~18 features per pair.**

**Why these specific features?**
- Different features catch different noise types. Jaro-Winkler is great for prefix matches. Token sort handles word reordering. Jaccard is robust to extra words. Numeric overlap catches address-number mismatches.
- The ML model learns which features matter most and how to combine them.

### Step 4: Model (Classifier)
**What:** Train a binary classifier: given the 18 features, predict **match (1)** or **no match (0)**.

**Algorithm: XGBoost** (gradient-boosted decision trees)

**Why XGBoost?**
| Why | Explanation |
|-----|-------------|
| Handles mixed feature types | Works with both continuous (similarity scores) and binary (flags) features |
| Feature interactions | Trees naturally capture "name matches AND address matches → likely match" |
| Fast | Histogram-based training, parallelizable |
| Robust | Not sensitive to feature scaling, handles missing values |
| Proven | Dominant in tabular ML competitions |

**Alternative models:**
| Model | When to consider |
|-------|-----------------|
| LightGBM | Faster than XGBoost on very large data, similar quality |
| Random Forest | Simpler, less prone to overfitting, but usually worse |
| Logistic Regression | Baseline, fast, but can't capture feature interactions |
| Neural Network / Transformer | Could work with raw text, but overkill here and slower |
| Siamese Network | End-to-end text matching, but needs a lot of data |

**Training data creation:**
- **Positive pairs:** (S1, S2/S3) pairs from the ground truth.
- **Negative pairs:** (S1, S2/S3) pairs that came from blocking but are NOT in the ground truth. These are **hard negatives** — they looked similar enough to pass blocking but aren't actual matches. This is much better than random negatives.
- **Negative ratio:** 5:1 or higher. Since F0.5 is precision-heavy, we want the model to see many negatives and learn to say "no" confidently.

**Output:** Probability (0–1) that a pair is a match. Then apply a **threshold** to decide.

### Step 5: Threshold Selection
**What:** Choose the probability cutoff. Above threshold → match. Below → no match.

**Why it matters for F0.5:**
- High threshold (e.g., 0.8) → very few matches → high precision, low recall
- Low threshold (e.g., 0.3) → many matches → low precision, high recall
- F0.5 **punishes false positives 2× more than false negatives**, so the optimal threshold is usually **higher** (0.4–0.7 range).

**How to find it:**
Sweep thresholds from 0.15 to 0.85 on a validation set. For each threshold, compute F0.5. Pick the one that gives the highest F0.5.

---

## 6. Hyperparameters — What, Why, How to Tune

### XGBoost Hyperparameters

| Parameter | What it does | Typical range | Why it matters |
|-----------|-------------|---------------|----------------|
| `max_depth` | Max tree depth | 3–10 | Deeper = more complex, risk overfitting |
| `learning_rate` | Step size per tree | 0.01–0.3 | Lower = more trees needed but better generalization |
| `n_estimators` | Number of trees | 100–1000 | More trees + low learning rate = better, but slower |
| `subsample` | Fraction of rows per tree | 0.5–1.0 | <1 adds randomness, reduces overfitting |
| `colsample_bytree` | Fraction of features per tree | 0.5–1.0 | Same as subsample but for features |
| `min_child_weight` | Min samples in a leaf | 1–20 | Higher = more conservative (good for precision) |
| `scale_pos_weight` | Weight for positive class | 1–10 | Higher = model tries harder to find positives |
| `gamma` | Min loss reduction to split | 0–5 | Higher = more pruning = simpler trees |
| `reg_alpha` | L1 regularization | 1e-8–10 | Sparsity in feature usage |
| `reg_lambda` | L2 regularization | 1e-8–10 | Penalizes large weights |

### How to Tune

**Option A: Manual (fast, good enough)**
Start with defaults. Adjust `max_depth` (try 5, 7, 9), `learning_rate` (try 0.05, 0.1), and `scale_pos_weight` (try 1, 3, 5). Pick whatever gives best validation F0.5.

**Option B: Optuna (automated, better)**
Use Optuna (Bayesian optimization). It:
1. Tries random params initially.
2. Builds a model of which params work.
3. Focuses on promising regions.
4. Reports best combo after N trials.

```python
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # ... other params
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    # Evaluate F0.5 on validation set
    return f05_score  # Optuna maximizes this

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

**Option C: Grid search (slow, exhaustive)**
Try every combination. Don't do this — too slow with 10 hyperparameters.

### Blocking Hyperparameters
| Parameter | What | Typical value |
|-----------|------|---------------|
| `top_k` | Max candidates per S1 entity | 30–100 |
| `ngram_range` | Char n-gram size for TF-IDF | (2,4) or (3,5) |
| `max_features` | TF-IDF vocabulary cap | 50K–100K |

Higher `top_k` = better recall but more pairs to score (slower).

---

## 7. Evaluation — F0.5 Score

### The Formula

$$F_{0.5} = \frac{1.25 \times Precision \times Recall}{0.25 \times Precision + Recall}$$

### Per-Entity Calculation

For each S1 entity, compare your predicted matches vs ground truth:

```
Predicted: {S2-00047, S2-00193, S3-00812}
Truth:     {S2-00047, S3-00812}

TP = 2 (S2-00047, S3-00812 — correct)
FP = 1 (S2-00193 — wrong match)
FN = 0 (nothing missed)

Precision = 2/3 = 0.667
Recall    = 2/2 = 1.0
F0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
```

### Singleton Handling
| Predicted | Truth | Score | Why |
|-----------|-------|-------|-----|
| empty | empty | **1.0** | Correctly said "no matches" |
| non-empty | empty | **0.0** | False merge — worst case |
| empty | non-empty | **0.0** | Missed everything |

### Macro Average
F0.5 is computed **per S1 entity**, then **averaged** across ALL S1 entities. Every entity counts equally.

### Why Precision-Heavy?
In real business data, **merging two different businesses is catastrophic** (wrong invoices, legal issues). Missing a link is just inconvenient. So F0.5 weights precision 2× over recall.

**Practical implication:** Be conservative. It's better to miss a match than to predict a wrong one.

---

## 8. Output Format

### `matching_results.tsv` (scored on leaderboard)
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```
- One row per S1 test entity. **Every** S1 entity must appear.
- Empty `matched_entity_ids` = singleton (no matches).
- Only S2/S3 IDs allowed. No S1 IDs.
- No duplicate IDs within a row.
- Tab-separated, not comma-separated file.

### `candidate_pairs.tsv` (not scored, but required)
Same format but with `candidate_entity_ids` — the broader set before your model filtered.

### Validation Script
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Run this before submitting. It catches formatting errors.

---

## 9. Constraints
- Model must be **MIT/Apache 2.0 licensed** and **≤ 8 billion parameters**. XGBoost is fine.
- **No external data** — no APIs, no geocoding, no business databases. Only the provided data.
- **No external lookup** of any kind.

---

## 10. Summary — What Makes or Breaks Your Score

| Factor | Impact | Why |
|--------|--------|-----|
| Blocking recall | **Critical** | If a true match isn't in candidates, you can never find it |
| Feature quality | **High** | Better features = easier job for the classifier |
| Threshold tuning | **High** | F0.5 is very sensitive to threshold. Too low = false merges tank your score |
| Singleton handling | **Medium** | ~50%+ of entities may be singletons. Getting these right is free points |
| Hyperparameter tuning | **Medium** | Optuna gets you a few % improvement over defaults |
| Preprocessing | **Medium** | Without normalization, string similarity features are weaker |
| Model choice | **Low** | XGBoost vs LightGBM doesn't matter much. Features matter more |
