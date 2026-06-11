"""
Response Accuracy Scorer
Computes accuracy between actual bot response and expected reference answer.
"""

import re


def tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return set(text.split())


def keyword_accuracy(actual: str, expected_keywords: list) -> float:
    """
    Compute what fraction of expected keywords appear in the actual response.

    Args:
        actual           (str):  The bot's actual response.
        expected_keywords(list): List of keyword strings to look for.

    Returns:
        float: 0.0 – 100.0 (percentage of keywords found).
    """
    if not expected_keywords or not actual:
        return 0.0
    actual_lower = actual.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in actual_lower)
    return round((found / len(expected_keywords)) * 100, 1)


def reference_accuracy(actual: str, reference: str) -> float:
    """
    Token-level F1 overlap between actual and reference answer.

    Args:
        actual    (str): The bot's actual response.
        reference (str): The ideal reference answer.

    Returns:
        float: 0.0 – 100.0 F1 accuracy score.
    """
    if not actual or not reference:
        return 0.0

    actual_tokens    = tokenize(actual)
    reference_tokens = tokenize(reference)

    if not reference_tokens:
        return 0.0

    overlap   = actual_tokens & reference_tokens
    precision = len(overlap) / len(actual_tokens)    if actual_tokens    else 0
    recall    = len(overlap) / len(reference_tokens) if reference_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 100, 1)


def score_response(actual: str, expected_keywords: list, reference: str = "") -> dict:
    """
    Full accuracy report for a single test case.

    Returns:
        dict with keys: keyword_accuracy, reference_accuracy, overall_accuracy,
                        matched_keywords, total_keywords.
    """
    kw_score  = keyword_accuracy(actual, expected_keywords)
    ref_score = reference_accuracy(actual, reference) if reference else kw_score

    actual_lower = actual.lower()
    matched  = [kw for kw in expected_keywords if kw.lower() in actual_lower]
    missed   = [kw for kw in expected_keywords if kw.lower() not in actual_lower]

    # Overall = weighted average (60% keywords, 40% reference)
    if reference:
        overall = round(kw_score * 0.6 + ref_score * 0.4, 1)
    else:
        overall = kw_score

    return {
        "keyword_accuracy":   kw_score,
        "reference_accuracy": ref_score,
        "overall_accuracy":   overall,
        "matched_keywords":   matched,
        "missed_keywords":    missed,
        "total_keywords":     len(expected_keywords),
    }
