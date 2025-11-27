#!/bin/bash

# Fine-tuned RAG LLM Evaluation Script
# This script evaluates a fine-tuned LLM on multiple subject CSV files using RAG workflow

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
MODEL_PATH="${PROJECT_ROOT}/model/finetuned_rag_llm/251127"  # Path to fine-tuned RAG model
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"  # Path to model directory for infer_top_k
MAX_NEW_TOKENS=20
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=2600
TOP_K=20
NUM_SAMPLES=200
NUM_EXAMPLES=0
INFER_DEVICE="cuda"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Fine-tuned RAG LLM Evaluation ===${NC}"

# Check if fine-tuned model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Fine-tuned model directory not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please run rag_finetuning_llm.sh first to train the model.${NC}"
    exit 1
fi

# Check if tool model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Tool model directory not found: $MODEL_DIR${NC}"
    echo -e "${YELLOW}Please make sure the achievement_classifier best model exists.${NC}"
    exit 1
fi

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "Fine-tuned model path: ${YELLOW}${MODEL_PATH}${NC}"
echo -e "Tool model dir: ${YELLOW}${MODEL_DIR}${NC}"
echo -e "Device: ${YELLOW}${DEVICE}${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max input length: ${YELLOW}${MAX_INPUT_LENGTH}${NC}"
echo -e "Num samples: ${YELLOW}${NUM_SAMPLES}${NC}"
echo -e "Top-k: ${YELLOW}${TOP_K}${NC}"
echo -e "Infer device: ${YELLOW}${INFER_DEVICE}${NC}"
echo -e "Num examples: ${YELLOW}${NUM_EXAMPLES}${NC}"
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
    SUBJECT="${BASENAME%.csv}"
    SUBJECT_TRAIN_CSV="${PROJECT_ROOT}/dataset/train_80/${SUBJECT}.csv"
    
    if [ ! -f "$SUBJECT_TRAIN_CSV" ]; then
        echo -e "${RED}Error: Train CSV not found for ${SUBJECT}: ${SUBJECT_TRAIN_CSV}${NC}"
        ((FAILED++))
        continue
    fi
    
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing: ${BASENAME}${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
    
    # Prepare command arguments
    CMD_ARGS=(
        --input_csv "$CSV_FILE"
        --model_path "$MODEL_PATH"
        --train-csv "$SUBJECT_TRAIN_CSV"
        --model-dir "$MODEL_DIR"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --device "$DEVICE"
        --max-input-length "$MAX_INPUT_LENGTH"
        --top-k "$TOP_K"
        --infer-device "$INFER_DEVICE"
        --num-samples "$NUM_SAMPLES"
        --num-examples "$NUM_EXAMPLES"
    )
    
    # Run fine-tuned RAG LLM evaluation
    if $PYTHON_CMD "${PROJECT_ROOT}/src/rag_llm_text_classification/rag_eval_ft_llm.py" "${CMD_ARGS[@]}"; then
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
echo -e "${GREEN}║  Fine-tuned RAG LLM Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/rag_llm_text_classification/finetuned_results_*.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/rag_llm_text_classification/finetuned_logs/${NC}"
