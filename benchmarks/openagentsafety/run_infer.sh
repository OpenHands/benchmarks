#!/usr/bin/env bash
set -eo pipefail

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Default values
DATASET="mgulavani/openagentsafety_full"
SPLIT="train"
MODEL="" #FIXME
OUTPUT_DIR="./eval_out"
MAX_ITERATIONS=100
EVAL_N_LIMIT=1
EVAL_NOTE="initial"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run OpenAgentSafety evaluation with specified parameters."
    echo
    echo "Options:"
    echo "  -n, --num-tasks NUM      Number of tasks to evaluate (default: $EVAL_N_LIMIT)"
    echo "  -m, --model MODEL        Model to use (default: $MODEL)"
    echo "  -o, --output DIR         Output directory (default: $OUTPUT_DIR)"
    echo "  -d, --dataset NAME       Dataset name (default: $DATASET)"
    echo "  -s, --split SPLIT        Dataset split (default: $SPLIT)"
    echo "  --max-iter NUM           Maximum iterations (default: $MAX_ITERATIONS)"
    echo "  --note TEXT             Evaluation note (default: $EVAL_NOTE)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 -n 5                  # Run 5 tasks"
    echo "  $0 -m custom/model-name  # Use specific model"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-tasks)
            EVAL_N_LIMIT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --note)
            EVAL_NOTE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            ;;
    esac
done

# Print configuration
echo "Environment:"
echo "  LITELLM_API_KEY: ${LITELLM_API_KEY:0:8}... (truncated)"
echo "  LITELLM_BASE_URL: $LITELLM_BASE_URL"
echo
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Split: $SPLIT"
echo "  Model: $MODEL"
echo "  Tasks: $EVAL_N_LIMIT"
echo "  Output: $OUTPUT_DIR"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Note: $EVAL_NOTE"
echo

# Run evaluation
export PYTHONPATH=/home/ubuntu/benchmarks:$PYTHONPATH
python3 -m benchmarks.openagentsafety.run_infer \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-iterations "$MAX_ITERATIONS" \
    --eval-n-limit "$EVAL_N_LIMIT" \
    --eval-note "$EVAL_NOTE"
