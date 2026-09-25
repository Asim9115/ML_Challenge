"""
Preprocessing — normalize business names and addresses.

Handles:
- Case folding
- Unicode / transliteration (Hindi Devanagari, French accents)
- Legal suffix expansion (Corp→Corporation, Pvt→Private, etc.)
- Address abbreviation expansion (Rd→Road, St→Street, etc.)
- Punctuation stripping
- Whitespace collapsing
"""

import re
import pandas as pd
from unidecode import unidecode

# ── Legal suffix map ──────────────────────────────────
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

# ── Address abbreviation map ─────────────────────────
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
    "nagar": "nagar",
    "dist": "district",
    "nr": "near",
    "opp": "opposite",
    "no": "number",
    "kh": "khasra",
}

# Compile regex for word boundary replacement
_LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in LEGAL_SUFFIXES) + r")\b\.?"
)
_ADDR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREVS) + r")\b\.?"
)


def _clean_base(text: str) -> str:
    """Lowercase, transliterate, strip punctuation, collapse whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = unidecode(text)  # Devanagari → latin, accents → ascii
    # Replace & with "and"
    text = text.replace("&", " and ")
    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize a business name."""
    text = _clean_base(name)
    if not text:
        return ""
    # Expand legal suffixes
    text = _LEGAL_PATTERN.sub(lambda m: LEGAL_SUFFIXES[m.group(1)], text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """Normalize a business address."""
    text = _clean_base(address)
    if not text:
        return ""
    # Expand address abbreviations
    text = _ADDR_PATTERN.sub(lambda m: ADDRESS_ABBREVS[m.group(1)], text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numeric_tokens(text: str) -> set:
    """Extract numeric tokens (zip codes, street numbers, etc.)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def get_combined_text(name_norm: str, addr_norm: str) -> str:
    """Combine name + address for TF-IDF. Name weighted by repetition."""
    # Repeat name to give it more weight in TF-IDF
    parts = []
    if name_norm:
        parts.append(name_norm)
        parts.append(name_norm)  # double-weight name
    if addr_norm:
        parts.append(addr_norm)
    return " ".join(parts)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a source DataFrame.

    Adds: name_norm, addr_norm, combined_text
    """
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("").str.strip()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["combined_text"] = df.apply(
        lambda r: get_combined_text(r["name_norm"], r["addr_norm"]), axis=1
    )
    return df


if __name__ == "__main__":
    # Quick sanity check
    tests = [
        ("Pvt. Ltd. Corp", "123 Main St., Apt 4B"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI"),
        ("Café René & Fils", "14 Rue de la Paix, Paris"),
        ("B+ Retail Inc", "1712 Montebello Ave, Phoenix, AZ"),
    ]
    for name, addr in tests:
        print(f"Name: {name!r}")
        print(f"  → {normalize_name(name)!r}")
        print(f"Addr: {addr!r}")
        print(f"  → {normalize_address(addr)!r}")
        print()
