#!/bin/bash

# Naive Cosine Similarity Evaluation
# This script runs batch cosine similarity evaluation on subject CSV files

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set paths
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cosine Similarity Evaluation ===${NC}"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo -e "${RED}Error: Dataset folder not found: $DATASET_FOLDER${NC}"
    exit 1
fi

echo -e "Dataset folder: ${YELLOW}${DATASET_FOLDER}${NC}"
echo ""

# Run batch cosine similarity
echo -e "${GREEN}Running batch cosine similarity evaluation...${NC}"
python "${PROJECT_ROOT}/src/cosine_similarity/batch_cosine_similarity.py" --folder_path "$DATASET_FOLDER"

echo ""
echo -e "${GREEN}✓ Cosine similarity evaluation completed!${NC}"
echo -e "Results saved to: ${YELLOW}${PROJECT_ROOT}/output/cosine_similarity/results.json${NC}"
