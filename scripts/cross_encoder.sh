#!/bin/bash

# Cross Encoder Training and Evaluation
# This script fine-tunes and evaluates a cross encoder model

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
TRAIN_CSV="${PROJECT_ROOT}/dataset/training_subject_text20/과학.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/validation_subject_text20/과학.csv"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cross Encoder Training and Evaluation ===${NC}"

# Check if input CSVs exist
if [ ! -f "$TRAIN_CSV" ]; then
    echo -e "${RED}Error: Training CSV not found: $TRAIN_CSV${NC}"
    echo "Please provide a valid CSV file path"
    exit 1
fi

if [ ! -f "$VALIDATION_CSV" ]; then
    echo -e "${RED}Error: Validation CSV not found: $VALIDATION_CSV${NC}"
    echo "Please provide a valid CSV file path"
    exit 1
fi

echo -e "Training CSV: ${YELLOW}${TRAIN_CSV}${NC}"
echo -e "Validation CSV: ${YELLOW}${VALIDATION_CSV}${NC}"
echo -e "Model save path: ${YELLOW}${PROJECT_ROOT}/model/cross_finetuned${NC}"
echo ""

# Step 1: Fine-tune cross encoder
echo -e "${BLUE}[Step 1/2] Fine-tuning cross encoder...${NC}"
python "${PROJECT_ROOT}/src/cross_encoder/finetune_cross_encoder.py" --input_csv "$TRAIN_CSV"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Fine-tuning failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Fine-tuning completed!${NC}"
echo ""

# Step 2: Evaluate cross encoder
echo -e "${BLUE}[Step 2/2] Evaluating cross encoder...${NC}"
python "${PROJECT_ROOT}/src/cross_encoder/eval_cross_encoder.py" --input_csv "$VALIDATION_CSV"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Evaluation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Cross encoder training and evaluation completed!${NC}"
echo -e "${GREEN}Model saved to: ${YELLOW}${PROJECT_ROOT}/model/cross_finetuned${NC}"
echo -e "${GREEN}Results saved to: ${YELLOW}${PROJECT_ROOT}/output/cross_encoder/results_rerank.json${NC}"

