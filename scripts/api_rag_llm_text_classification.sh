#!/bin/bash

# API-based RAG LLM Text Classification Evaluation
# This script runs API-based RAG (Retrieval-Augmented Generation) LLM classification evaluation on subject CSV files

# set -e  # Exit on error (주석 처리: 에러가 나도 다음 파일 계속 처리)

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"

# API Configuration
API_PROVIDER="openrouter"
API_MODEL="qwen/qwen-2.5-7b-instruct:free"
API_DELAY=1.0
# API_KEY .env에서

# RAG Configuration
TOP_K=20
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"
INFER_DEVICE="cuda"

# Evaluation Configuration
MAX_NEW_TOKENS=20
TEMPERATURE=0.1
MAX_TOTAL_SAMPLES=200
FEW_SHOT=True
NUM_EXAMPLES=5

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== API-based RAG LLM Text Classification Evaluation ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Retrieval model directory not found: $MODEL_DIR${NC}"
    echo -e "${YELLOW}Please make sure the achievement_classifier best model exists.${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "API Provider: ${YELLOW}${API_PROVIDER}${NC}"
echo -e "API Model: ${YELLOW}${API_MODEL}${NC}"
echo -e "API Delay: ${YELLOW}${API_DELAY}s${NC}"
echo -e "Max new tokens: ${YELLOW}${MAX_NEW_TOKENS}${NC}"
echo -e "Temperature: ${YELLOW}${TEMPERATURE}${NC}"
echo -e "Max total samples: ${YELLOW}${MAX_TOTAL_SAMPLES}${NC}"
echo -e "Top-k: ${YELLOW}${TOP_K}${NC}"
echo -e "Retrieval model dir: ${YELLOW}${MODEL_DIR}${NC}"
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

    # Build few-shot flag
    if [ "$FEW_SHOT" = True ]; then
        FEW_SHOT_FLAG="--few-shot"
    else
        FEW_SHOT_FLAG=""
    fi

    # Run API RAG evaluation
    if python "${PROJECT_ROOT}/src/rag_llm_text_classification/rag_eval_llm.py" \
        --input_csv "$CSV_FILE" \
        --api-provider "$API_PROVIDER" \
        --api-model "$API_MODEL" \
        --api-delay "$API_DELAY" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num-samples "$MAX_TOTAL_SAMPLES" \
        --top-k "$TOP_K" \
        --train-csv "$SUBJECT_TRAIN_CSV" \
        --model-dir "$MODEL_DIR" \
        --infer-device "$INFER_DEVICE" \
        --num-examples "$NUM_EXAMPLES" \
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
echo -e "${GREEN}║  API RAG LLM Classification Evaluation Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo -e "Processed: ${GREEN}${PROCESSED}${NC} / Total: ${BLUE}${#CSV_FILES[@]}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}"
fi
echo ""
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/rag_llm_text_classification/results.json${NC}"
echo -e "Logs saved to: ${YELLOW}${PROJECT_ROOT}/output/rag_llm_text_classification/logs/${NC}"
