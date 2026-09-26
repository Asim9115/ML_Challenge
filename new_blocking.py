"""
Memory-safe, checkpointed blocking pipeline for the entity resolution task.
Designed for 12GB RAM Colab.

v3 -- root cause fix + recall tuning, based on real diagnostic results:
  * CRITICAL FIX: country field is now normalized (strip + lower) before
    partitioning. Root-cause check found a record with country="US" in
    source1 and country="US " (trailing space) in source2 -- these were
    silently routed into DIFFERENT country buckets and could never match,
    even with jaccard=1.0 identical text. This alone likely explains a large
    chunk of the 0.40 recall.
  * RECALL FIX: diagnostic on 15 real missed matches found 15/15 were
    "cutoff" misses (signal present, just ranked below the old top_k cutoff),
    0/15 were structural. Avg candidates/row was 46.6 against a hard cap of
    60 -- real headroom existed. Raised:
      TOP_K_NGRAM   40  -> 120
      TOP_K_SNM     20  -> 60
      SNM_WINDOW    15  -> 50
    This roughly doubles per-row compute but you have headroom (candidates
    stage was running at ~10 sec/chunk of 20k rows, well within budget).
  * All params overridable via CLI without editing the file.

BECAUSE THE COUNTRY-PARTITIONING LOGIC CHANGED, ANY PREVIOUSLY BUILT INDEX
IS NOW STALE AND MUST BE REBUILT. Wipe blocking_work/ and output/ before
rerunning stage index, or pass --reset to do it for you.

RUN
---
python blocking_pipeline.py --split train --stage index --reset
python blocking_pipeline.py --split train --stage candidates

Optional tuning flags (all have the new defaults above if omitted):
  --top-k-ngram INT
  --top-k-snm INT
  --snm-window INT
  --max-postings INT
"""

import os
import gc
import json
import pickle
import shutil
import argparse
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# CONFIG (defaults; all overridable via CLI flags below)
# --------------------------------------------------------------------------
DATA_DIR = "/content/dataset/preprocessed"
WORK_DIR = "blocking_work"
OUT_DIR = "output"

NGRAM_N = 3
TOP_K_NGRAM_DEFAULT = 120
TOP_K_SNM_DEFAULT = 60
SNM_WINDOW_DEFAULT = 50
MAX_POSTINGS_PER_NGRAM_DEFAULT = 5000
S1_CHUNK_SIZE = 20_000

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
PROGRESS_FILE = os.path.join(WORK_DIR, "progress.json")


# --------------------------------------------------------------------------
# PROGRESS / CHECKPOINTING
# --------------------------------------------------------------------------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"index_built": {}, "candidates_done": {}}


def save_progress(progress):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def char_ngrams(s, n=NGRAM_N):
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def sort_key(s):
    """Token-sorted prefix key: robust to word reordering."""
    if not s:
        return ""
    toks = sorted(s.split())
    return "".join(toks)[:12]


def normalize_country(c):
    """
    CRITICAL: strip + lowercase so 'US', 'US ', 'us', ' US' all partition
    together. Without this, whitespace/case noise in the raw country field
    silently splits true matches into different, non-comparable buckets --
    this was the root cause of the 0.40 recall on the real data.
    """
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return ""
    return str(c).strip().lower()


def clean_text_cols(chunk):
    chunk["name_clean"] = chunk["name_clean"].fillna("").astype(str)
    chunk["addr_clean"] = chunk["addr_clean"].fillna("").astype(str)
    chunk["country"] = chunk["country"].apply(normalize_country)
    return chunk


def get_countries(split):
    path = os.path.join(DATA_DIR, f"{split}_source1_clean.tsv")
    countries = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=["country"], chunksize=500_000):
        norm = chunk["country"].apply(normalize_country)
        countries.update(c for c in norm.unique().tolist() if c)
    return sorted(countries)


def safe_folder_name(country):
    return country.replace(" ", "_").replace("/", "_") or "unknown"


# --------------------------------------------------------------------------
# STAGE 1: BUILD INVERTED INDEX (per country, S2+S3 combined)
# --------------------------------------------------------------------------
def build_index_for_country(split, country, max_postings):
    idx_path = os.path.join(WORK_DIR, f"index_{split}_{safe_folder_name(country)}.npz_dir")
    os.makedirs(idx_path, exist_ok=True)

    int_id = 0
    id_map = {}
    inv_index = defaultdict(list)
    sort_keys = []

    for src in ["source2", "source3"]:
        path = os.path.join(DATA_DIR, f"{split}_{src}_clean.tsv")
        cols = ["entity_id", "name_clean", "addr_clean", "country"]
        for chunk in pd.read_csv(path, sep="\t", usecols=cols, chunksize=200_000):
            chunk = clean_text_cols(chunk)          # normalize BEFORE filtering
            chunk = chunk[chunk["country"] == country]
            if chunk.empty:
                continue

            for eid, name, addr in chunk[["entity_id", "name_clean", "addr_clean"]].itertuples(index=False):
                id_map[int_id] = eid
                text = f"{name} {addr}".strip()
                grams = set(char_ngrams(text))
                for g in grams:
                    inv_index[g].append(int_id)
                sort_keys.append((sort_key(name), int_id))
                int_id += 1
            del chunk
            gc.collect()

    for g in list(inv_index.keys()):
        if len(inv_index[g]) > max_postings:
            del inv_index[g]

    sort_keys.sort(key=lambda t: t[0])
    sorted_ids = np.array([t[1] for t in sort_keys], dtype=np.int64)
    sorted_key_strs = np.array([t[0] for t in sort_keys], dtype=object)

    pd.Series(id_map).to_pickle(os.path.join(idx_path, "id_map.pkl"))
    with open(os.path.join(idx_path, "inv_index.pkl"), "wb") as f:
        pickle.dump(dict(inv_index), f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(idx_path, "sorted_ids.npy"), sorted_ids)
    np.save(os.path.join(idx_path, "sorted_keys.npy"), sorted_key_strs, allow_pickle=True)

    n_docs = int_id
    del id_map, inv_index, sort_keys, sorted_ids
    gc.collect()
    return idx_path, n_docs


def stage_build_index(split, max_postings):
    progress = load_progress()
    countries = get_countries(split)
    print(f"[{split}] normalized countries found: {countries}")

    for country in countries:
        key = f"{split}::{country}"
        if progress["index_built"].get(key):
            print(f"[skip] index already built for {key}")
            continue
        print(f"[build] index for {key} ...")
        idx_path, n_docs = build_index_for_country(split, country, max_postings)
        progress["index_built"][key] = {"path": idx_path, "n_docs": n_docs}
        save_progress(progress)
        print(f"[done] {key}: {n_docs} S2+S3 docs indexed -> {idx_path}")
        if n_docs == 0:
            print(f"[WARN] 0 docs indexed for {key} -- check that this country actually "
                  f"appears in source2/source3 with a matching normalized country value")


# --------------------------------------------------------------------------
# STAGE 2: GENERATE CANDIDATES (per country, chunked over S1)
# --------------------------------------------------------------------------
def load_index(idx_path):
    id_map = pd.read_pickle(os.path.join(idx_path, "id_map.pkl")).to_dict()
    with open(os.path.join(idx_path, "inv_index.pkl"), "rb") as f:
        inv_index = pickle.load(f)
    sorted_ids = np.load(os.path.join(idx_path, "sorted_ids.npy"))
    sorted_keys = np.load(os.path.join(idx_path, "sorted_keys.npy"), allow_pickle=True)
    return id_map, inv_index, sorted_ids, sorted_keys


def ngram_candidates(text, inv_index, top_k):
    grams = char_ngrams(text)
    if not grams:
        return []
    counts = Counter()
    for g in grams:
        postings = inv_index.get(g)
        if postings:
            counts.update(postings)
    if not counts:
        return []
    return [cid for cid, _ in counts.most_common(top_k)]


def stage_generate_candidates(split, top_k_ngram, top_k_snm, snm_window):
    progress = load_progress()
    countries = get_countries(split)
    out_path = os.path.join(OUT_DIR, f"candidate_pairs_{split}.tsv")

    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    for country in countries:
        idx_key = f"{split}::{country}"
        if idx_key not in progress["index_built"]:
            print(f"[warn] no index for {idx_key}, run --stage index first. skipping.")
            continue
        idx_path = progress["index_built"][idx_key]["path"]
        id_map, inv_index, sorted_ids, sorted_keys = load_index(idx_path)

        s1_path = os.path.join(DATA_DIR, f"{split}_source1_clean.tsv")
        cols = ["entity_id", "name_clean", "addr_clean", "country"]

        chunk_idx = 0
        for chunk in pd.read_csv(s1_path, sep="\t", usecols=cols, chunksize=S1_CHUNK_SIZE):
            chunk = clean_text_cols(chunk)           # normalize BEFORE filtering
            chunk = chunk[chunk["country"] == country]
            if chunk.empty:
                chunk_idx += 1
                continue

            chunk_key = f"{split}::{country}::chunk{chunk_idx}"
            if progress["candidates_done"].get(chunk_key):
                chunk_idx += 1
                continue

            rows_out = []
            n = len(sorted_ids)

            for eid, name, addr in chunk[["entity_id", "name_clean", "addr_clean"]].itertuples(index=False):
                text = f"{name} {addr}".strip()
                ng_cands = ngram_candidates(text, inv_index, top_k_ngram)

                key_str = sort_key(name)
                pos = int(np.searchsorted(sorted_keys, key_str))
                lo = max(0, pos - snm_window)
                hi = min(n, pos + snm_window)
                snm_cands = sorted_ids[lo:hi].tolist()[:top_k_snm]

                union_ids = set(ng_cands) | set(snm_cands)
                candidate_eids = [id_map[i] for i in union_ids]
                rows_out.append((eid, ",".join(candidate_eids)))

            pd.DataFrame(rows_out, columns=["source1_entity_id", "candidate_entity_ids"]) \
                .to_csv(out_path, sep="\t", mode="a", header=False, index=False)

            progress["candidates_done"][chunk_key] = True
            save_progress(progress)
            print(f"[checkpoint] {chunk_key}: {len(rows_out)} S1 rows written")

            del chunk, rows_out
            gc.collect()
            chunk_idx += 1

        del id_map, inv_index, sorted_ids, sorted_keys
        gc.collect()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--stage", required=True, choices=["index", "candidates"])
    parser.add_argument("--top-k-ngram", type=int, default=TOP_K_NGRAM_DEFAULT)
    parser.add_argument("--top-k-snm", type=int, default=TOP_K_SNM_DEFAULT)
    parser.add_argument("--snm-window", type=int, default=SNM_WINDOW_DEFAULT)
    parser.add_argument("--max-postings", type=int, default=MAX_POSTINGS_PER_NGRAM_DEFAULT)
    parser.add_argument("--reset", action="store_true",
                         help="Wipe blocking_work/ and output/ before running "
                              "(required once after upgrading to this version, "
                              "since the country-partitioning fix invalidates "
                              "any previously built index)")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(WORK_DIR):
            shutil.rmtree(WORK_DIR)
        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)
        os.makedirs(WORK_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        print("[reset] wiped blocking_work/ and output/")

    if args.stage == "index":
        stage_build_index(args.split, args.max_postings)
    else:
        stage_generate_candidates(args.split, args.top_k_ngram, args.top_k_snm, args.snm_window)

    print("Done. Progress saved to", PROGRESS_FILE)