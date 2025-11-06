#!/bin/bash

# API-based LLM Text Classification Evaluation
# This script runs API-based LLM classification evaluation on subject CSV files

# set -e  # Exit on error (주석 처리: 에러가 나도 다음 파일 계속 처리)

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"

# API Configuration
API_PROVIDER="openrouter"  # Options: openai, anthropic, google, openrouter
API_MODEL="qwen/qwen3-14b:free"  # Provider-specific model name
API_DELAY=1.0  # Delay in seconds between API calls (to avoid rate limits)
# API_KEY will be loaded from .env automatically

# Evaluation Configuration
MAX_NEW_TOKENS=20
TEMPERATURE=0.1
MAX_CANDIDATES=15
MAX_TOTAL_SAMPLES=100
MAX_SAMPLES_PER_ROW=5
FEW_SHOT=False

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== API-based LLM Text Classification Evaluation ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "API Provider: ${YELLOW}${API_PROVIDER}${NC}"
echo -e "API Model: ${YELLOW}${API_MODEL}${NC}"
echo -e "API Delay: ${YELLOW}${API_DELAY}s${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max total samples: ${YELLOW}${MAX_TOTAL_SAMPLES}${NC}"
echo -e "Max candidates: ${YELLOW}${MAX_CANDIDATES}${NC}"
echo ""

# Get list of CSV files (정렬된 순서로)
CSV_FILES=($(find "$DATASET_FOLDER" -name "*.csv" -type f | sort))

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No CSV files found in $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "${BLUE}Found ${#CSV_FILES[@]} CSV files to process${NC}"
echo -e "${YELLOW}Files to process:${NC}"
for file in "${CSV_FILES[@]}"; do
    echo -e "  - $(basename "$file")"
done
echo ""

# Process each CSV file
PROCESSED=0
FAILED=0

for CSV_FILE in "${CSV_FILES[@]}"; do
    BASENAME=$(basename "$CSV_FILE")
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing: ${BASENAME}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
    
    # Build few-shot flag
    if [ "$FEW_SHOT" = True ]; then
        FEW_SHOT_FLAG="--few-shot"
    else
        FEW_SHOT_FLAG=""
    fi
    
    # Run API LLM evaluation
    if python "${PROJECT_ROOT}/src/llm_text_classification/eval_llm.py" \
        --input_csv "$CSV_FILE" \
        --api-provider "$API_PROVIDER" \
        --api-model "$API_MODEL" \
        --api-delay "$API_DELAY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --max-total-samples "$MAX_TOTAL_SAMPLES" \
        --max-candidates "$MAX_CANDIDATES" \
        --max-samples-per-row "$MAX_SAMPLES_PER_ROW" \
        $FEW_SHOT_FLAG; then
        echo -e "${GREEN}✓ Successfully processed ${BASENAME}${NC}"
        ((PROCESSED++))
    else
        echo -e "${RED}✗ Failed to process ${BASENAME}${NC}"
        ((FAILED++))
    fi
    echo ""
done

# Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  API LLM Classification Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/results.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/logs/${NC}"

