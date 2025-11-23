#!/bin/bash

# LLM-based Text Classification Evaluation
# This script runs LLM-based classification evaluation on subject CSV files

# set -e  # Exit on error (주석 처리: 에러가 나도 다음 파일 계속 처리)

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Determine Python command (python3 if python is not available)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}Error: Neither python nor python3 found${NC}"
    exit 1
fi

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_NAME="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_NEW_TOKENS=10
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=2000
TOP_K=20
MAX_TOTAL_SAMPLES=200
MAX_SAMPLES_PER_ROW=2
FEW_SHOT=True
TRAIN_CSV="${PROJECT_ROOT}/dataset/train.csv"
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"
INFER_DEVICE="cuda"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Agentic LLM Text Classification Evaluation ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

# Check if train CSV exists
if [ ! -f "$TRAIN_CSV" ]; then
    echo -e "${RED}Error: Train CSV file not found: $TRAIN_CSV${NC}"
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Tool model directory not found: $MODEL_DIR${NC}"
    echo -e "${YELLOW}Please make sure the achievement_classifier best model exists.${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "Model: ${YELLOW}${MODEL_NAME}${NC}"
echo -e "Device: ${YELLOW}${DEVICE}${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max input length: ${YELLOW}${MAX_INPUT_LENGTH}${NC}"
echo -e "Max total samples: ${YELLOW}${MAX_TOTAL_SAMPLES}${NC}"
echo -e "Top-k: ${YELLOW}${TOP_K}${NC}"
echo -e "Train CSV: ${YELLOW}${TRAIN_CSV}${NC}"
echo -e "Tool model dir: ${YELLOW}${MODEL_DIR}${NC}"
echo -e "Infer device: ${YELLOW}${INFER_DEVICE}${NC}"
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
    
    # Run LLM evaluation
    if [ "$FEW_SHOT" = True ]; then
        FEW_SHOT_FLAG="--few-shot"
    else
        FEW_SHOT_FLAG=""
    fi
    if $PYTHON_CMD "${PROJECT_ROOT}/src/agentic_llm_text_classification/agentic_eval_llm.py" \
        --input_csv "$CSV_FILE" \
        --model_name "$MODEL_NAME" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --device "$DEVICE" \
        --max-input-length "$MAX_INPUT_LENGTH" \
        --max-total-samples "$MAX_TOTAL_SAMPLES" \
        --top-k "$TOP_K" \
        --max-samples-per-row "$MAX_SAMPLES_PER_ROW" \
        --train-csv "$TRAIN_CSV" \
        --model-dir "$MODEL_DIR" \
        --infer-device "$INFER_DEVICE" \
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
echo -e "${GREEN}║  Agentic LLM Classification Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/results.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/llm_text_classification/logs/${NC}"

