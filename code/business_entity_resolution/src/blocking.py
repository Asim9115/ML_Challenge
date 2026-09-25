"""
Blocking — candidate generation.

Reduces the O(N*M) comparison space to a manageable candidate set.
Every true match must survive blocking or recall is capped.

Outputs: for each S1 entity, a list of candidate S2/S3 entity_ids.
"""

# TODO: implement
# Strategies to explore:
# - TF-IDF on name+address → cosine top-K
# - Phonetic blocking keys (Soundex, Metaphone on name tokens)
# - Token/n-gram overlap blocking
# - Country-based partitioning (only compare within same country)
# - Multi-pass blocking (union of candidates from multiple strategies)
#
# Must be fast enough at scale: ~2M S1 × ~10M S2+S3
