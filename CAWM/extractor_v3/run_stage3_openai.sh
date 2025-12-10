#!/bin/bash
# Run Stage 3 with official OpenAI API instead of CMU Gateway
# Make sure to set your OpenAI API key: export OPENAI_API_KEY=sk-...

set -e

OUTPUT_BASE="${1:-/home/tsljgj/private/benchmarks/CAWM/extractor_v3_output}"
MODEL="${MODEL:-gpt-4o}"  # Using GPT-4o instead of gpt-5

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY=sk-..."
    exit 1
fi

WORKFLOW_FILE="$OUTPUT_BASE/stage2_workflows/all_workflows.json"
if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "Error: Stage 2 output not found at: $WORKFLOW_FILE"
    exit 1
fi

echo "============================================"
echo "Stage 3: Combine Workflows (OpenAI API)"
echo "============================================"
echo "Input file: $WORKFLOW_FILE"
echo "Model: $MODEL"
echo "Using: Official OpenAI API"
echo ""

# Run with empty base-url to use official OpenAI API
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage3_combine_workflows.py \
    --workflow-file "$WORKFLOW_FILE" \
    --output-file "$OUTPUT_BASE/final_workflows.json" \
    --model "$MODEL" \
    --base-url "" \
    --api-key "$OPENAI_API_KEY"

echo ""
echo "============================================"
echo "Stage 3 Complete!"
echo "============================================"
