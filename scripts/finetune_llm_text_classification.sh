#!/bin/bash

# Fine-tuned LLM Evaluation Script
# This script evaluates a fine-tuned LLM on multiple subject CSV files

# set -e  # Exit on error (주석 처리: 에러가 나도 다음 파일 계속 처리)

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_PATH="${PROJECT_ROOT}/model/finetuned_llm"  # Path to fine-tuned model
MAX_NEW_TOKENS=50
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=6144
MAX_CANDIDATES=120
MAX_TOTAL_SAMPLES=100

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Fine-tuned LLM Evaluation ===${NC}"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please run finetuning_llm.sh first to train the model.${NC}"
    exit 1
fi

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "Model path: ${YELLOW}${MODEL_PATH}${NC}"
echo -e "Device: ${YELLOW}${DEVICE}${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max input length: ${YELLOW}${MAX_INPUT_LENGTH}${NC}"
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
    
    # Prepare command arguments
    CMD_ARGS=(
        --input_csv "$CSV_FILE"
        --model_path "$MODEL_PATH"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --device "$DEVICE"
        --max-input-length "$MAX_INPUT_LENGTH"
        --max-total-samples "$MAX_TOTAL_SAMPLES"
        --max-candidates "$MAX_CANDIDATES"
    )
    
    # Run fine-tuned LLM evaluation
    if python "${PROJECT_ROOT}/src/llm_text_classification/eval_finetune_llm.py" "${CMD_ARGS[@]}"; then
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
echo -e "${GREEN}║  Fine-tuned LLM Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/finetuned_results.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/finetuned_logs/${NC}"

