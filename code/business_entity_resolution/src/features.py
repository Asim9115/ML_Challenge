"""
Feature engineering — build similarity features for each candidate pair.

Input:  (S1_entity, S2/S3_candidate) pairs from blocking.
Output: feature matrix ready for the classifier.
"""

# TODO: implement
# Name features:
# - Jaccard similarity (token-level)
# - Levenshtein / edit distance (normalized)
# - Jaro-Winkler
# - TF-IDF cosine similarity
# - Exact match flag
# - Common token count / ratio
#
# Address features:
# - Token overlap ratio
# - Edit distance (normalized)
# - PIN/ZIP code match flag
# - State/city match flag
#
# Other:
# - Country exact match
# - Name length ratio
# - Address length ratio
