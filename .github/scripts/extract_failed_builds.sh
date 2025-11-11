#!/bin/bash
# Extract failed base images from manifest.jsonl files
# Usage: extract_failed_builds.sh <output_file>
#
# This script:
# 1. Finds all manifest.jsonl files in the builds/ directory
# 2. Extracts base images that failed (have errors or no tags)
# 3. Writes unique failed images to the specified output file
# 4. Returns 0 if there are failures, 1 if all builds succeeded

set -euo pipefail

OUTPUT_FILE="${1:-}"

if [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <output_file>"
    exit 2
fi

# Find all manifest.jsonl files
MANIFEST_FILES=$(find builds -name "manifest.jsonl" -type f 2>/dev/null || true)

if [ -z "$MANIFEST_FILES" ]; then
    echo "No manifest.jsonl files found"
    exit 1
fi

# Extract failed base images
FAILED_IMAGES=$(cat $MANIFEST_FILES | python3 -c "
import sys
import json
for line in sys.stdin:
    data = json.loads(line.strip())
    if data.get('error') is not None or len(data.get('tags', [])) == 0:
        print(data.get('base_image', ''))
" | sort -u)

if [ -z "$FAILED_IMAGES" ]; then
    echo "No failed builds found"
    exit 1
else
    FAILED_COUNT=$(echo "$FAILED_IMAGES" | wc -l)
    echo "Found $FAILED_COUNT failed builds"
    echo "$FAILED_IMAGES" > "$OUTPUT_FILE"
    echo "Saved failed images to $OUTPUT_FILE"
    exit 0
fi
