#!/bin/bash
# Run only Stage 3: Combine workflows
# This script is useful when Stage 2 has already completed successfully

set -e  # Exit on error

# Configuration
OUTPUT_BASE="${1:-/home/tsljgj/private/benchmarks/CAWM/extractor_v3_output}"
MODEL="${MODEL:-gpt-5}"
BASE_URL="${BASE_URL:-https://ai-gateway.andrew.cmu.edu/}"
API_KEY="${OPENAI_API_KEY}"

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# Verify stage 2 output exists
WORKFLOW_FILE="$OUTPUT_BASE/stage2_workflows/all_workflows.json"
if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "Error: Stage 2 output not found at: $WORKFLOW_FILE"
    echo "Please run stage 2 first or specify correct OUTPUT_BASE path"
    exit 1
fi

echo "============================================"
echo "Stage 3: Combine Workflows"
echo "============================================"
echo "Input file: $WORKFLOW_FILE"
echo "Output directory: $OUTPUT_BASE"
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"
echo ""

# Stage 3: Combine workflows
echo "Stage 3: Combining similar workflows..."
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage3_combine_workflows.py \
    --workflow-file "$WORKFLOW_FILE" \
    --output-file "$OUTPUT_BASE/final_workflows.json" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY"

echo ""
echo "============================================"
echo "Stage 3 Complete!"
echo "============================================"
echo "Final workflows saved to: $OUTPUT_BASE/final_workflows.json"
echo ""
echo "Output files:"
echo "  - Unfiltered: $OUTPUT_BASE/final_workflows_unfiltered.json"
echo "  - Filtered: $OUTPUT_BASE/final_workflows.json"
echo "  - Plain text: $OUTPUT_BASE/final_workflows.txt"
