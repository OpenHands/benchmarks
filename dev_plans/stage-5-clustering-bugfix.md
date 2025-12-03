# Stage 5: Clustering Algorithm Bug Fix

## Problem

41 trajectories → 1 cluster → only 10 experiences extracted

With `action_sequence` clustering, ALL trajectories were being grouped into a single cluster because the `_calc_seq_similarity` function used `set()` which loses sequence order information.

## Root Cause Analysis

### Bug Location
`CAWM/clustering.py:139-149`

### Old Code (Buggy)
```python
def _calc_seq_similarity(self, t1: Trajectory, t2: Trajectory) -> float:
    """Jaccard similarity of action types."""
    s1 = set(t1.get_action_sequence())  # BUG: loses order!
    s2 = set(t2.get_action_sequence())

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union
```

### Why It's Wrong

All SWE-bench trajectories have similar action type SETS:
- `{TESTING, FILE_VIEW, OTHER, EXPLORATION, THINK, NAVIGATION, FILE_EDIT}`

Using `set()` ignores:
1. **Order**: A→B→C is treated the same as C→B→A
2. **Frequency**: A→A→A→B is treated the same as A→B

Result: Jaccard similarity = 0.875 - 1.000 for ALL pairs!

```
Traj 1 vs Traj 2: 0.875 (SAME CLUSTER)
Traj 1 vs Traj 3: 1.000 (SAME CLUSTER)
Traj 1 vs Traj 4: 1.000 (SAME CLUSTER)
...
```

## Solution

### New Code (Fixed)
```python
def _calc_seq_similarity(self, t1: Trajectory, t2: Trajectory) -> float:
    """
    Calculate sequence similarity using n-gram Jaccard similarity.

    Instead of just using set of action types (which loses order),
    we use n-grams (bigrams and trigrams) to capture sequential patterns.
    """
    seq1 = [a.name for a in t1.get_action_sequence()]
    seq2 = [a.name for a in t2.get_action_sequence()]

    # Generate n-grams (bigrams and trigrams)
    def get_ngrams(seq: list, n: int) -> set:
        if len(seq) < n:
            return set()
        return {tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)}

    # Use combination of bigrams and trigrams for pattern matching
    bigrams1 = get_ngrams(seq1, 2)
    bigrams2 = get_ngrams(seq2, 2)
    trigrams1 = get_ngrams(seq1, 3)
    trigrams2 = get_ngrams(seq2, 3)

    # Combine all n-grams
    ngrams1 = bigrams1 | trigrams1
    ngrams2 = bigrams2 | trigrams2

    # Jaccard similarity of n-grams
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return intersection / union
```

### Why N-grams Work

N-grams preserve sequence patterns:
- Bigram: `(EXPLORATION, FILE_VIEW)` captures "explored then viewed file"
- Trigram: `(FILE_EDIT, TESTING, FILE_EDIT)` captures "edit-test-edit cycle"

Different trajectories have different n-gram signatures even if their action type SETS are identical.

## Additional Changes

### Added `REPOSITORY` Clustering
```python
class SimilarityMethod(Enum):
    PROBLEM_DESCRIPTION = "problem_description"
    ACTION_SEQUENCE = "action_sequence"
    CODE_MODIFICATION = "code_modification"
    REPOSITORY = "repository"  # NEW: Group by repository
    RANDOM = "random"
```

This clusters by repository (astropy, django, etc.) for domain-specific experiences.

## Results

### Clustering Comparison (threshold=0.7)

| Method | Before Fix | After Fix |
|--------|------------|-----------|
| Clusters | 1 | 20 |
| Experiences | 10 | 75 |

### Threshold Sensitivity (action_sequence)

| Threshold | Clusters |
|-----------|----------|
| 0.3 | 1 |
| 0.5 | 2 |
| 0.7 | 20 |
| 0.8 | 36 |

### Sample Experiences (After Fix)

1. "When NDDataRef mask propagation fails with handle_mask=np.bitwise_or..."
2. "When django.utils.http.parse_http_date incorrectly interprets two-digit years..."
3. "When inner class references fail during migration serialization → Use __qualname__ instead of __name__"
4. "When PostgreSQL dbshell fails with additional parameters after database name..."

## Files Modified

| File | Change |
|------|--------|
| `CAWM/clustering.py` | Fixed `_calc_seq_similarity()` to use n-gram similarity |
| `CAWM/clustering.py` | Added `SimilarityMethod.REPOSITORY` enum |
| `CAWM/clustering.py` | Added `_cluster_by_repository()` method |
| `CAWM/main.py` | Added "repository" to CLI clustering options |

## Recommended Settings

For diverse experience extraction:
```bash
# Action sequence with threshold 0.7 (20 clusters from 41 trajectories)
OPENROUTER_API_KEY="..." uv run python CAWM/main.py \
    --output CAWM/workflow/output \
    --clustering action_sequence \
    --threshold 0.7
```
