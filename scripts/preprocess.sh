#!/bin/bash

# KorEduBench Data Preprocessing Script
# This script processes the Curriculum-Level Subject Dataset from AI Hub

set -e  # Exit on error

# Configuration
BASE_DIR="/mnt/e/2025_2_KorEduBench"
MAX_TEXTS=20
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

# Process function for each dataset type
process_dataset() {
    local DATASET_TYPE=$1
    local LABEL_DIR="${BASE_DIR}/${DATASET_TYPE}/label"
    local prefix=$(echo "$DATASET_TYPE" | tr '[:upper:]' '[:lower:]')
    
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Processing ${DATASET_TYPE} Dataset${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
    echo -e "Label Directory: ${YELLOW}${LABEL_DIR}${NC}"
    echo ""
    
    # Check if label directory exists
    if [ ! -d "$LABEL_DIR" ]; then
        echo -e "${RED}Error: Label directory not found at ${LABEL_DIR}${NC}"
        return 1
    fi
    
    # Step 1: Extract unique achievement standards
    echo -e "${GREEN}[Step 1/3] Extracting unique achievement standards...${NC}"
    python extract_standards.py "$LABEL_DIR" "${prefix}_unique_achievement_standards.csv"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Standards extracted successfully${NC}"
    else
        echo -e "${RED}✗ Failed to extract standards${NC}"
        return 1
    fi
    echo ""
    
    # Step 2: Add text samples to each standard
    echo -e "${GREEN}[Step 2/3] Adding text samples to standards...${NC}"
    python add_text_to_standards.py "$LABEL_DIR" \
        --csv_path "${prefix}_unique_achievement_standards.csv" \
        --output_csv "${prefix}_text_achievement_standards.csv" \
        --max_texts "$MAX_TEXTS"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Text samples added successfully${NC}"
    else
        echo -e "${RED}✗ Failed to add text samples${NC}"
        return 1
    fi
    echo ""
    
    # Step 3: Split CSV by subject
    echo -e "${GREEN}[Step 3/3] Splitting dataset by subject...${NC}"
    # Create output directory with dataset type prefix
    local OUTPUT_DIR="${prefix}_subject_text${MAX_TEXTS}"
    python split_subject.py \
        --input "${prefix}_text_achievement_standards.csv" \
        --max-texts "$MAX_TEXTS" \
        --encoding "utf-8-sig"
    
    # Rename the output directory to include prefix
    if [ -d "subject_text${MAX_TEXTS}" ]; then
        rm -rf "$OUTPUT_DIR"  # Remove if exists
        mv "subject_text${MAX_TEXTS}" "$OUTPUT_DIR"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset split by subject successfully${NC}"
    else
        echo -e "${RED}✗ Failed to split dataset${NC}"
        return 1
    fi
    echo ""
    
    echo -e "${GREEN}✓ ${DATASET_TYPE} dataset preprocessing complete!${NC}"
    echo -e "${GREEN}Generated files:${NC}"
    echo -e "  - ${prefix}_unique_achievement_standards.csv"
    echo -e "  - ${prefix}_text_achievement_standards.csv"
    echo -e "  - ${OUTPUT_DIR}/ (directory with subject-specific CSV files)"
    echo ""
}

# Process both Training and Validation datasets
process_dataset "Training"
echo ""
process_dataset "Validation"
echo ""

# Final Summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  All Preprocessing Complete!${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Generated datasets:${NC}"
echo -e "  ${YELLOW}Training:${NC}"
echo -e "    - training_unique_achievement_standards.csv"
echo -e "    - training_text_achievement_standards.csv"
echo -e "    - training_subject_text${MAX_TEXTS}/"
echo ""
echo -e "  ${YELLOW}Validation:${NC}"
echo -e "    - validation_unique_achievement_standards.csv"
echo -e "    - validation_text_achievement_standards.csv"
echo -e "    - validation_subject_text${MAX_TEXTS}/"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Run cosine similarity on training: cd cosine_similarity && python batch_cosine_similarity.py --folder_path ../dataset/training_subject_text${MAX_TEXTS}"
echo -e "  2. Run cosine similarity on validation: cd cosine_similarity && python batch_cosine_similarity.py --folder_path ../dataset/validation_subject_text${MAX_TEXTS}"
echo -e "  3. Finetune cross encoder: cd cross_encoder && python finetune_cross_encoder.py --input_csv {csv_path}"

