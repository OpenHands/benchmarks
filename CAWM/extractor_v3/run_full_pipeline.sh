#!/bin/bash
# Run the complete workflow extraction pipeline using GPT-5

set -e  # Exit on error

# Configuration
TRAJECTORY_FILE="${1:-/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl}"
OUTPUT_BASE="${2:-/home/tsljgj/private/benchmarks/CAWM/extractor_v3_output}"
LIMIT="${3:-10}"
MODEL="${MODEL:-gpt-5}"
BASE_URL="${BASE_URL:-https://ai-gateway.andrew.cmu.edu/}"
API_KEY="${OPENAI_API_KEY}"

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

echo "============================================"
echo "Workflow Extraction Pipeline (GPT-5)"
echo "============================================"
echo "Trajectory file: $TRAJECTORY_FILE"
echo "Output directory: $OUTPUT_BASE"
echo "Limit: $LIMIT trajectories"
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/stage0_cleaned"
mkdir -p "$OUTPUT_BASE/stage1_clusters"
mkdir -p "$OUTPUT_BASE/stage2_workflows"

# Stage 0: Clean trajectories
echo "Stage 0: Cleaning trajectories..."
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage0_clean.py \
    --input-file "$TRAJECTORY_FILE" \
    --output-dir "$OUTPUT_BASE/stage0_cleaned" \
    --limit "$LIMIT" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY"

echo ""
echo "Stage 0 complete!"
echo ""

# Stage 1: Cluster problems
echo "Stage 1: Clustering problems..."
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage1_cluster.py \
    --problem-file "$OUTPUT_BASE/stage0_cleaned/problem_descriptions.json" \
    --output-file "$OUTPUT_BASE/stage1_clusters/clusters.json" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY"

echo ""
echo "Stage 1 complete!"
echo ""

# Stage 2: Extract workflows
echo "Stage 2: Extracting workflows..."
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage2_extract_workflows.py \
    --cluster-file "$OUTPUT_BASE/stage1_clusters/clusters.json" \
    --trajectory-dir "$OUTPUT_BASE/stage0_cleaned" \
    --output-dir "$OUTPUT_BASE/stage2_workflows" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY"

echo ""
echo "Stage 2 complete!"
echo ""

# Stage 3: Combine workflows
echo "Stage 3: Combining similar workflows..."
python3 /home/tsljgj/private/benchmarks/CAWM/extractor_v3/stage3_combine_workflows.py \
    --workflow-file "$OUTPUT_BASE/stage2_workflows/all_workflows.json" \
    --output-file "$OUTPUT_BASE/final_workflows.json" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY"

echo ""
echo "Stage 3 complete!"
echo ""

echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo "Final workflows saved to: $OUTPUT_BASE/final_workflows.json"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_BASE/stage0_cleaned/        - Cleaned trajectory text files"
echo "  $OUTPUT_BASE/stage1_clusters/       - Problem clusters"
echo "  $OUTPUT_BASE/stage2_workflows/      - Workflows per cluster"
echo "  $OUTPUT_BASE/final_workflows.json   - Combined final workflows"
