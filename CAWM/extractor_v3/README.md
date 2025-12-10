# Workflow Extraction Pipeline

Extracts reusable workflows from Django repository agent trajectories.

## Quick Start

```bash
cd /home/tsljgj/private/benchmarks/CAWM

# Test data extraction (no API required)
./extractor_v3/QUICK_TEST.sh

# Run full pipeline (requires API key)
./extractor_v3/run_full_pipeline.sh
```

## What's New

### 1. Stage 0 - Better Detail Preservation
- Preserves EXACT terminal commands
- 2-5 sentence reasoning (not too brief)
- Abstracts non-Django elements as `{{variable_name}}`
- Keeps Django files concrete (`run_tests.py`, `manage.py`, etc.)
- **NEW**: Error handling - no more empty files!

### 2. Stage 2 - Type 1 Format Change
**Before:**
```json
{"steps": [{"reasoning": "...", "action": "..."}]}
```

**After:**
```json
{"steps": ["Step 1: ...", "Step 2: ...", ...]}
```

### 3. Trajectory Tracking
All workflows now track source trajectories:
```json
{
  "name": "Debugging Django Models",
  "steps": [...],
  "source_trajectories": ["django__django-12470", "django__django-12125"]
}
```

### 4. Multi-Source Filtering
Stage 3 generates 3 files:
- `final_workflows_unfiltered.json` - All workflows
- `final_workflows.json` - Filtered (2+ sources only)
- `final_workflows.txt` - Plain text version

## Output Structure

```
extractor_v3_output/
├── stage0_cleaned/
│   ├── django__django-*.txt              # Cleaned trajectories
│   └── problem_descriptions.json
├── stage1_clusters/
│   └── clusters.json
├── stage2_workflows/
│   └── all_workflows.json
└── final_workflows_unfiltered.json
    final_workflows.json                   # Multi-source only
    final_workflows.txt                    # Human-readable
```

## Error Handling

- **Empty LLM responses**: Falls back to simple format
- **API failures**: Creates fallback content instead of empty files
- **Missing data**: Skips trajectory with warning

## Documentation

- **[CHANGES.md](CHANGES.md)** - Detailed changes
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing instructions
- **[QUICK_TEST.sh](QUICK_TEST.sh)** - Automated tests
