# Stage 4: Experience Extraction Fix - Complete

## Summary

Fixed the CAWM pipeline to extract **task-specific experiences** instead of generic sequential workflows.

## Problem

Before: 41 trajectories → 1 cluster → 6 generic sequential steps (useless)
- "First reproduce the bug"
- "Write tests before fixing"
- "Read the code carefully"

## Solution

After: 41 trajectories → 1 cluster → **10 task-specific experiences**
- "When NDData mask is None, add explicit null check before bitwise operations"
- "Use __qualname__ instead of __name__ for inner class serialization in migrations"
- "For two-digit year parsing, implement RFC 7231: 00-69 → 2000-2069, 70-99 → 1970-1999"

## Changes Made

### 1. `CAWM/induction.py` - Prompt Rewrite (Lines 79-147)

**Old prompt**: "Extract HIGH-LEVEL, GENERAL workflows"

**New prompt**:
```
Extract SPECIFIC EXPERIENCES and LESSONS LEARNED...
- Each experience should be a CONCRETE, ACTIONABLE insight
- NOT generic steps like "write tests" or "read the code"
- Focus on WHAT was learned, not the process of learning it

Good examples:
- "When encountering 'FieldError: Cannot resolve keyword', check if the field name conflicts with a reverse relation"
- "For HTTP date parsing issues, ensure RFC 7231 compliance"

Bad examples (too generic):
- "First reproduce the bug" (obvious)
- "Write tests before fixing" (generic advice)
```

### 2. `CAWM/induction.py` - Parsing Logic (Lines 149-214)

Added support for new "experiences" format:
```json
{
    "experiences": [
        {
            "trigger": "When/If <specific situation>",
            "insight": "The key discovery",
            "action": "Specific fix",
            "category": "debugging|refactoring|testing|configuration|api_usage"
        }
    ]
}
```

### 3. `CAWM/pipeline.py` - Default Config (Lines 19-29)

```python
clustering_method: SimilarityMethod = SimilarityMethod.PROBLEM_DESCRIPTION  # was ACTION_SEQUENCE
clustering_threshold: float = 0.2  # was 0.5
llm_model: str = "moonshotai/kimi-k2-0905"  # was claude-3.5-sonnet
```

### 4. `CAWM/main.py` - CLI Defaults (Lines 88-101)

Updated default clustering to `problem_description` with threshold `0.2`.

## Output Example

```
1. When NDData arithmetic fails with mask propagation and one operand lacks a mask
   → The root cause is that None masks are not properly handled in bitwise operations
   → Check if either operand's mask is None before applying bitwise operations

2. When parsing two-digit years in HTTP dates and getting incorrect century interpretation
   → RFC 7231 specifies that years 00-69 should be 2000-2069, 70-99 → 1970-1999
   → Implement proper two-digit year conversion logic

3. When makemigrations fails to serialize inner class references correctly
   → The serializer uses __name__ which only gives the class name
   → Use __qualname__ instead of __name__ for inner classes
```

## Future Improvements

1. **Better Clustering**: Use embeddings or LLM-based semantic clustering to create more diverse clusters
2. **Batch Processing**: Process trajectories in smaller batches for more granular experiences
3. **Repository-Specific**: Cluster by repository (astropy, django, etc.) for domain-specific experiences

## Verification

```bash
OPENROUTER_API_KEY="..." uv run python CAWM/main.py --output CAWM/workflow/testing --verbose
```

Result: 10 task-specific experiences extracted from 41 trajectories.
