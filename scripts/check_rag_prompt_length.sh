#!/bin/bash

# Prompt Length Checker for RAG LLM Classification
# This script checks prompt token lengths for both Step 1 (query generation) and Step 2 (final selection)
# across all CSV files in a folder

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
TOP_K=20
NUM_SAMPLES=200
FEW_SHOT=True
PRINT_SAMPLE_PROMPT=True

OUTPUT_PATH="${PROJECT_ROOT}/output/check_rag_prompt_length/results_${TOP_K}_${NUM_SAMPLES}_${FEW_SHOT}.csv"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RAG Prompt Length Checker ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo -e "Model (tokenizer): ${YELLOW}${MODEL_NAME}${NC}"
echo -e "Top-k candidates: ${YELLOW}${TOP_K}${NC}"
echo -e "Num samples: ${YELLOW}${NUM_SAMPLES}${NC}"
echo -e "Output path: ${YELLOW}${OUTPUT_PATH}${NC}"
echo ""

# Run prompt length checker
echo -e "${BLUE}Running RAG prompt length checker...${NC}"
echo -e "${BLUE}(Checking Step 1 and Step 2 prompts)${NC}"
echo ""

if python "${PROJECT_ROOT}/src/test/check_rag_prompt_length.py" \
    --input_dir "$DATASET_FOLDER" \
    --model_name "$MODEL_NAME" \
    --top-k "$TOP_K" \
    --num-samples "$NUM_SAMPLES" \
    --output_path "$OUTPUT_PATH" \
    --print-sample-prompt "$PRINT_SAMPLE_PROMPT" \
    --few-shot "$FEW_SHOT"; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  RAG Prompt Length Check Complete!${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
    echo -e "Results saved to: ${YELLOW}${OUTPUT_PATH}${NC}"
else
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  RAG Prompt Length Check Failed!${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
