"""
new_preprocess.py — Preprocess all raw TSVs once and save cleaned versions to disk.

Run once:
    python new_preprocess.py

After this, every other script loads from preprocessed/ — no re-cleaning needed.

What it does per row:
  1. Lowercase
  2. Transliterate (Hindi/French → ASCII via unidecode)
  3. Replace & → and
  4. Strip punctuation
  5. Expand name abbreviations (Corp→Corporation, Pvt→Private, etc.)
  6. Expand address abbreviations (Rd→Road, St→Street, etc.)
  7. Collapse whitespace
  8. Create combined_text (name + address merged for TF-IDF later)

Output columns: entity_id, business_name, business_address, country, name_clean, addr_clean, combined_text
"""

import os
import re
import time
import pandas as pd
from unidecode import unidecode
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
OUT_DIR = os.path.join(DATA_DIR, "preprocessed")

FILES = [
    ("train", "train_source1.tsv"),
    ("train", "train_source2.tsv"),
    ("train", "train_source3.tsv"),
    ("test",  "test_source1.tsv"),
    ("test",  "test_source2.tsv"),
    ("test",  "test_source3.tsv"),
]

# ─── Abbreviation Maps ───────────────────────────────────

LEGAL_SUFFIXES = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pvt": "private",
    "co": "company",
    "assoc": "associates",
    "intl": "international",
    "natl": "national",
    "govt": "government",
    "grp": "group",
    "svcs": "services",
    "svc": "service",
    "mfg": "manufacturing",
    "mgmt": "management",
    "dept": "department",
    "univ": "university",
    "hosp": "hospital",
    "tech": "technology",
    "ind": "industries",
    "ent": "enterprises",
    "sys": "systems",
    "soln": "solutions",
    "sols": "solutions",
    "engr": "engineering",
    "engg": "engineering",
    "infra": "infrastructure",
    "pharma": "pharmaceutical",
    "fdn": "foundation",
    "ctr": "center",
}

ADDRESS_ABBREVS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "pkwy": "parkway",
    "hwy": "highway",
    "sq": "square",
    "trl": "trail",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "po": "post office",
    "mt": "mount",
    "ft": "fort",
    "jn": "junction",
    "expy": "expressway",
    "ext": "extension",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "dist": "district",
    "nr": "near",
    "opp": "opposite",
    "no": "number",
    "kh": "khasra",
}

# Pre-compile regex patterns (sorted by length desc so longer matches first)
_legal_keys = sorted(LEGAL_SUFFIXES.keys(), key=len, reverse=True)
_addr_keys = sorted(ADDRESS_ABBREVS.keys(), key=len, reverse=True)

_LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _legal_keys) + r")\b\.?"
)
_ADDR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _addr_keys) + r")\b\.?"
)


# ─── Cleaning Functions ──────────────────────────────────

def _clean_base(text):
    """Lowercase → strip leading junk → transliterate → strip punctuation → collapse whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    # Strip leading junk chars: dashes, dots, hashes, slashes, asterisks, underscores
    text = re.sub(r"^[\-\.\#\/\*\_\~\|\:\;\,\!\?\s]+", "", text)
    text = unidecode(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name(name):
    """Clean a business name: base clean + expand legal suffixes."""
    text = _clean_base(name)
    if not text:
        return ""
    text = _LEGAL_PATTERN.sub(lambda m: LEGAL_SUFFIXES[m.group(1)], text)
    return re.sub(r"\s+", " ", text).strip()


def clean_address(address):
    """Clean a business address: base clean + expand address abbreviations."""
    text = _clean_base(address)
    if not text:
        return ""
    text = _ADDR_PATTERN.sub(lambda m: ADDRESS_ABBREVS[m.group(1)], text)
    return re.sub(r"\s+", " ", text).strip()


def make_combined(name_clean, addr_clean):
    """Merge name + address into one string for TF-IDF. Name repeated for extra weight."""
    parts = []
    if name_clean:
        parts.append(name_clean)
        parts.append(name_clean)  # double weight on name
    if addr_clean:
        parts.append(addr_clean)
    return " ".join(parts)


# ─── Main ─────────────────────────────────────────────────

def preprocess_file(split, filename):
    """Read one raw TSV, clean it, save to preprocessed/."""
    in_path = os.path.join(DATA_DIR, split, filename)
    out_name = filename.replace(".tsv", "_clean.tsv")
    out_path = os.path.join(OUT_DIR, out_name)

    print(f"\n{'='*60}")
    print(f"Processing: {split}/{filename}")
    print(f"  Input:  {in_path}")
    print(f"  Output: {out_path}")

    t0 = time.time()

    # Read
    print("  Reading...", end=" ", flush=True)
    df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"{len(df):,} rows loaded in {time.time()-t0:.1f}s")

    # Fill empty strings
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("").str.strip()

    # Clean name
    print("  Cleaning names...", flush=True)
    t1 = time.time()
    tqdm.pandas(desc="    name_clean")
    df["name_clean"] = df["business_name"].progress_apply(clean_name)
    print(f"    Done in {time.time()-t1:.1f}s")

    # Clean address
    print("  Cleaning addresses...", flush=True)
    t1 = time.time()
    tqdm.pandas(desc="    addr_clean")
    df["addr_clean"] = df["business_address"].progress_apply(clean_address)
    print(f"    Done in {time.time()-t1:.1f}s")

    # Combined text
    print("  Creating combined_text...", end=" ", flush=True)
    t1 = time.time()
    df["combined_text"] = df.apply(
        lambda r: make_combined(r["name_clean"], r["addr_clean"]), axis=1
    )
    print(f"Done in {time.time()-t1:.1f}s")

    # Save
    print(f"  Saving to {out_path}...", end=" ", flush=True)
    t1 = time.time()
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Done in {time.time()-t1:.1f}s")

    # Quick stats
    total_time = time.time() - t0
    empty_names = (df["name_clean"] == "").sum()
    empty_addrs = (df["addr_clean"] == "").sum()
    print(f"  Stats: {len(df):,} rows, {empty_names:,} empty names, {empty_addrs:,} empty addresses")
    print(f"  Total time: {total_time:.1f}s")

    # Show a few examples
    print(f"\n  Sample (first 3 rows):")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"    [{row['entity_id']}]")
        print(f"      name:  {row['business_name']!r}")
        print(f"        →    {row['name_clean']!r}")
        print(f"      addr:  {row['business_address']!r}")
        print(f"        →    {row['addr_clean']!r}")


def main():
    print("=" * 60)
    print("PREPROCESSING ALL DATASETS")
    print(f"Output folder: {OUT_DIR}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    total_start = time.time()

    for split, filename in FILES:
        preprocess_file(split, filename)

    print(f"\n{'='*60}")
    print(f"ALL DONE in {time.time()-total_start:.1f}s")
    print(f"Cleaned files saved to: {OUT_DIR}")
    print(f"\nFiles created:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".tsv"):
            size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
