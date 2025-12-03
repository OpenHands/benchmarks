#!/usr/bin/env python3
"""
Minimal script to reproduce the HuggingFace authentication error with GAIA dataset.

This script demonstrates the two places where authentication is required:
1. Loading the dataset metadata
2. Downloading the dataset files

Prerequisites:
    pip install datasets huggingface-hub

Run without HF_TOKEN to see the error:
    python reproduce_gaia_auth_error.py

Run with HF_TOKEN to succeed:
    HF_TOKEN=hf_xxxxx python reproduce_gaia_auth_error.py

Or use uv:
    uv run --with datasets --with huggingface-hub python reproduce_gaia_auth_error.py
"""

import os
import sys


print("=" * 70)
print("GAIA Dataset Authentication Test")
print("=" * 70)

# Check if HF_TOKEN is set
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    print(f"✓ HF_TOKEN is set: {hf_token[:10]}...")
else:
    print("✗ HF_TOKEN is NOT set")
    print("  Expected: Dataset access will fail")

print("-" * 70)

# Test 1: Load dataset metadata
print("\n[Test 1] Loading GAIA dataset metadata...")
print("Location: benchmarks/gaia/run_infer.py:72")
print("Code: load_dataset('gaia-benchmark/GAIA', '2023_level1')")
print()

try:
    from datasets import DatasetDict, load_dataset

    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_level1",
        token=hf_token,
    )
    print("✓ SUCCESS: Dataset metadata loaded")
    # Cast to DatasetDict since load_dataset with a config returns a dict of splits
    assert isinstance(dataset, DatasetDict)
    print(f"  Splits available: {list(dataset.keys())}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}")
    print(f"  Error: {str(e)}")
    print()
    print("This is the EXPECTED error without authentication!")
    sys.exit(1)

print("-" * 70)

#
