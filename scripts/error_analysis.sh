#!/bin/bash

# Error Analysis Script
# This script runs error analysis on RAG LLM text classification results

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Determine Python command (python3 if python is not available)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Neither python nor python3 found"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Change to project root directory
cd "$PROJECT_ROOT"

# Define logs directories to process
declare -a LOGS_DIRS=(
    "output/rag_finetuned_llm_text_classification/unsloth_Qwen2.5-7B-Instruct-bnb-4bit_25-11-28/logs"
    "output/rag_llm_text_classification/openrouter_qwen_qwen3-8b_25-11-28/logs"
    "output/rag_llm_text_classification/openrouter_qwen_qwen3-8b_25-11-28_0/logs"
    "output/rag_llm_text_classification/openrouter_qwen_qwen3-next-80b-a3b-instruct_25-11-28_5/logs"
    "output/rag_llm_text_classification/openrouter_llama_llama-3.3-70B-Instruct/logs"
)

# If argument is provided, use it instead
if [ $# -gt 0 ]; then
    LOGS_DIRS=("$@")
fi

# Process each logs directory
TOTAL=${#LOGS_DIRS[@]}
SUCCESS=0
FAILED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running error analysis on $TOTAL directory(ies)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for i in "${!LOGS_DIRS[@]}"; do
    LOGS_DIR="${LOGS_DIRS[$i]}"
    NUM=$((i + 1))
    
    echo -e "${YELLOW}[$NUM/$TOTAL] Processing: $LOGS_DIR${NC}"
    
    # Check if logs directory exists
    if [ ! -d "$PROJECT_ROOT/$LOGS_DIR" ]; then
        echo -e "${RED}  Error: Logs directory not found: $PROJECT_ROOT/$LOGS_DIR${NC}"
        echo -e "${RED}  Skipping...${NC}"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi
    
    # Run error analysis
    $PYTHON_CMD src/test/error_analysis.py "$LOGS_DIR"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}  ✓ Completed successfully!${NC}"
        echo -e "${GREEN}  Output files:${NC}"
        echo -e "${GREEN}    - $LOGS_DIR/error_analysis.csv${NC}"
        echo -e "${GREEN}    - $LOGS_DIR/error_summary.csv${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}  ✗ Analysis failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary:${NC}"
echo -e "${GREEN}  Successful: $SUCCESS${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}  Failed: $FAILED${NC}"
fi
echo -e "${BLUE}========================================${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

