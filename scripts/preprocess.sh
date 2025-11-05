#!/bin/bash

# KorEduBench Data Preprocessing Script
# This script processes the Curriculum-Level Subject Dataset from AI Hub

set -e  # Exit on error

# Configuration
BASE_DIR="/mnt/e/2025_2_KorEduBench"
MAX_TEXTS=160
MIN_TEXTS=160
NUM_TEXTS=160
DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dataset"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== KorEduBench Preprocessing Script ===${NC}"
echo -e "Base Directory: ${YELLOW}${BASE_DIR}${NC}"
echo -e "Max Texts per Standard: ${YELLOW}${MAX_TEXTS}${NC}"
echo -e "Min Texts per Standard: ${YELLOW}${MIN_TEXTS}${NC}"
echo ""

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}Error: Base directory not found at ${BASE_DIR}${NC}"
    exit 1
fi

# Navigate to dataset directory
cd "$DATASET_DIR"
echo -e "${GREEN}Working directory: ${PWD}${NC}"
echo ""

# Step 1: Extract unique standards from Training dataset
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 1: Extract Unique Standards from Training${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
TRAIN_LABEL_DIR="${BASE_DIR}/Training/label"
echo -e "Label Directory: ${YELLOW}${TRAIN_LABEL_DIR}${NC}"

if [ ! -d "$TRAIN_LABEL_DIR" ]; then
    echo -e "${RED}Error: Training label directory not found at ${TRAIN_LABEL_DIR}${NC}"
    exit 1
fi

python ../src/preprocessing/extract_standards.py "$TRAIN_LABEL_DIR" "${DATASET_DIR}/unique_achievement_standards.csv"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Standards extracted successfully${NC}"
else
    echo -e "${RED}✗ Failed to extract standards${NC}"
    exit 1
fi
echo ""

# Step 2: Add text samples to each standard
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 2: Add Text Samples from Training${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/add_text_to_standards.py "$TRAIN_LABEL_DIR" \
    --csv_path "${DATASET_DIR}/unique_achievement_standards.csv" \
    --output_csv "${DATASET_DIR}/text_achievement_standards.csv" \
    --max_texts "$MAX_TEXTS"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Text samples added successfully${NC}"
else
    echo -e "${RED}✗ Failed to add text samples${NC}"
    exit 1
fi
echo ""

# Step 3: Verify not empty (optional)
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 3: Verify No Empty Data (Optional)${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/verify_not_empty.py "${DATASET_DIR}/text_achievement_standards.csv"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Verification passed${NC}"
else
    echo -e "${YELLOW}⚠ Verification warning (not critical)${NC}"
fi
echo ""

# Step 4: Check insufficient text
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 4: Check Insufficient Text${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/check_insufficient_text.py --min_texts "$MIN_TEXTS"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Insufficient text check completed${NC}"
else
    echo -e "${RED}✗ Failed to check insufficient text${NC}"
    exit 1
fi
echo ""

# Step 5: Add additional text samples from Validation dataset
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 5: Add Additional Text from Validation${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
VAL_LABEL_DIR="${BASE_DIR}/Validation/label"
echo -e "Label Directory: ${YELLOW}${VAL_LABEL_DIR}${NC}"

if [ ! -d "$VAL_LABEL_DIR" ]; then
    echo -e "${RED}Error: Validation label directory not found at ${VAL_LABEL_DIR}${NC}"
    exit 1
fi

python ../src/preprocessing/add_additional_text_to_standards.py "$VAL_LABEL_DIR" \
    --insufficient_csv "${DATASET_DIR}/insufficient_text.csv" \
    --text_standards_csv "${DATASET_DIR}/text_achievement_standards.csv" \
    --output_csv "${DATASET_DIR}/text_achievement_standards.csv" \
    --max_texts "$MAX_TEXTS"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Additional text samples added successfully${NC}"
else
    echo -e "${RED}✗ Failed to add additional text samples${NC}"
    exit 1
fi
echo ""

# Step 6: Check insufficient text again
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 6: Check Insufficient Text Again${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/check_insufficient_text.py --min_texts "$MIN_TEXTS"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Insufficient text check completed${NC}"
else
    echo -e "${RED}✗ Failed to check insufficient text${NC}"
    exit 1
fi
echo ""

# Step 7: Filter standards and split into train/valid
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 7: Filter Standards and Split into Train/Valid${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/filter_standards.py \
    --num_texts "$NUM_TEXTS" \
    --input_csv "${DATASET_DIR}/text_achievement_standards.csv" \
    --train_csv "${DATASET_DIR}/train.csv" \
    --valid_csv "${DATASET_DIR}/valid.csv"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Standards filtered and split successfully${NC}"
else
    echo -e "${RED}✗ Failed to filter standards${NC}"
    exit 1
fi
echo ""

# Step 8: Split train CSV by subject
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 8: Split Train Dataset by Subject${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
HALF_NUM_TEXTS=$((NUM_TEXTS / 2))
python ../src/preprocessing/split_subject.py \
    --input "${DATASET_DIR}/train.csv" \
    --output "train_text${HALF_NUM_TEXTS}" \
    --max-texts "$HALF_NUM_TEXTS" \
    --encoding "utf-8-sig"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Train dataset split by subject successfully${NC}"
else
    echo -e "${RED}✗ Failed to split train dataset${NC}"
    exit 1
fi
echo ""

# Step 9: Split valid CSV by subject
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Step 9: Split Valid Dataset by Subject${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
python ../src/preprocessing/split_subject.py \
    --input "${DATASET_DIR}/valid.csv" \
    --output "valid_text${HALF_NUM_TEXTS}" \
    --max-texts "$HALF_NUM_TEXTS" \
    --encoding "utf-8-sig"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Valid dataset split by subject successfully${NC}"
else
    echo -e "${RED}✗ Failed to split valid dataset${NC}"
    exit 1
fi
echo ""

# Final Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  All Preprocessing Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Generated datasets:${NC}"
echo -e "  - unique_achievement_standards.csv"
echo -e "  - text_achievement_standards.csv"
echo -e "  - insufficient_text.csv"
echo -e "  - train.csv"
echo -e "  - valid.csv"
echo -e "  - train_text${HALF_NUM_TEXTS}/ (directory with subject-specific train CSV files)"
echo -e "  - valid_text${HALF_NUM_TEXTS}/ (directory with subject-specific valid CSV files)"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Run cosine similarity on training: cd scripts && bash cosine_similarity.sh"
echo -e "  2. Finetune and evaluate cross encoder: cd scripts && bash cross_encoder.sh"

