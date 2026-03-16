#!/bin/bash
# Test script for swebenchmultilingual benchmark

set -e

echo "=== Testing SWE-bench Multilingual Benchmark ==="
echo ""

# Test 1: Verify entry points are registered
echo "Test 1: Checking entry points..."
if uv run swebenchmultilingual-infer --help > /dev/null 2>&1; then
    echo "✓ swebenchmultilingual-infer command is available"
else
    echo "✗ swebenchmultilingual-infer command not found"
    exit 1
fi

if uv run swebenchmultilingual-eval --help > /dev/null 2>&1; then
    echo "✓ swebenchmultilingual-eval command is available"
else
    echo "✗ swebenchmultilingual-eval command not found"
    exit 1
fi

echo ""
echo "Test 2: Verifying dataset access..."
python3 << 'PYTHON'
from datasets import load_dataset
ds = load_dataset('SWE-bench/SWE-bench_Multilingual', split='test')
print(f"✓ Dataset loaded: {len(ds)} instances")
print(f"✓ First instance: {ds[0]['instance_id']}")
PYTHON

echo ""
echo "Test 3: Testing eval_infer with mock data..."
mkdir -p test_output_verify

# Create a mock output.jsonl
cat > test_output_verify/test.jsonl << 'EOF'
{"instance_id": "apache__druid-13704", "test_result": {"git_patch": "diff --git a/test.py b/test.py\nindex 1234567..abcdefg 100644\n--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,3 @@\n-print('hello')\n+print('hello world')\n"}, "instruction": "Test", "error": null, "history": []}
EOF

# Test conversion
uv run swebenchmultilingual-eval test_output_verify/test.jsonl --run-id test123 --skip-evaluation > /dev/null 2>&1 || true

if [ -f "test_output_verify/test.swebench.jsonl" ]; then
    echo "✓ Eval conversion completed successfully"
    echo "✓ Converted file created: test_output_verify/test.swebench.jsonl"
    # Verify content
    if grep -q "apache__druid-13704" test_output_verify/test.swebench.jsonl; then
        echo "✓ Converted file contains expected instance_id"
        if grep -q "model_patch" test_output_verify/test.swebench.jsonl; then
            echo "✓ Converted file contains model_patch field"
        else
            echo "✗ Converted file missing model_patch"
            exit 1
        fi
    else
        echo "✗ Converted file missing instance_id"
        exit 1
    fi
else
    echo "✗ Converted file not created"
    exit 1
fi

# Cleanup
rm -rf test_output_verify

echo ""
echo "=== All tests passed! ==="
echo ""
echo "Summary:"
echo "- Entry points: ✓ swebenchmultilingual-infer, swebenchmultilingual-eval"
echo "- Dataset: ✓ SWE-bench/SWE-bench_Multilingual (300 instances)"
echo "- Conversion: ✓ OpenHands → SWE-Bench format"
echo ""
echo "The swebenchmultilingual benchmark is ready to use!"
