def _levenshtein_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = [list(range(n + 1))]
    for i in range(1, m + 1):
        row = [i]
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            row.append(min(dp[i - 1][j] + 1, row[j - 1] + 1, dp[i - 1][j - 1] + cost))
        dp.append(row)
    return dp[m][n]

def normalized_similarity(a: str, b: str) -> float:
    """Similarity 0-1 (higher = better match). Uses 1 - normalized Levenshtein distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein_distance(a, b)
    return 1.0 - (d / max(len(a), len(b)))