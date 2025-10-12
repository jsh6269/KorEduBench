#!/bin/bash

# LLM-based Text Classification Evaluation
# This script runs LLM-based classification evaluation on subject CSV files

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/validation_subject_text20"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS=200
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=2048
MAX_TOTAL_SAMPLES=100

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LLM Text Classification Evaluation ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "Model: ${YELLOW}${MODEL_NAME}${NC}"
echo -e "Device: ${YELLOW}${DEVICE}${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max input length: ${YELLOW}${MAX_INPUT_LENGTH}${NC}"
echo -e "Max total samples: ${YELLOW}${MAX_TOTAL_SAMPLES}${NC}"
echo ""

# Get list of CSV files
CSV_FILES=($(find "$DATASET_FOLDER" -name "*.csv" -type f))

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No CSV files found in $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "${BLUE}Found ${#CSV_FILES[@]} CSV files to process${NC}"
echo ""

# Process each CSV file
PROCESSED=0
FAILED=0

for CSV_FILE in "${CSV_FILES[@]}"; do
    BASENAME=$(basename "$CSV_FILE")
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing: ${BASENAME}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
    
    # Run LLM evaluation
    if python "${PROJECT_ROOT}/src/llm_text_classification/eval_llm.py" \
        --input_csv "$CSV_FILE" \
        --model_name "$MODEL_NAME" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --device "$DEVICE" \
        --max-input-length "$MAX_INPUT_LENGTH" \
        --max-total-samples "$MAX_TOTAL_SAMPLES"; then
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
echo -e "${GREEN}║  LLM Classification Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/results.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/logs/${NC}"

