#!/bin/bash

# LLM Fine-tuning Script
# This script fine-tunes an LLM on a training dataset and saves the model

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Training configuration
TRAIN_CSV="${PROJECT_ROOT}/dataset/training/train_data.csv"  # Update with your training data path
MODEL_NAME="unsloth/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_llm"
MAX_SEQ_LENGTH=6144
MAX_SAMPLES_PER_ROW=3     # Train with 3 samples per achievement standard (prevents overfitting)
MAX_TOTAL_SAMPLES=None    # No limit on total samples (after per-row filtering)
ENCODING="utf-8"

# Training hyperparameters
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
WARMUP_STEPS=5
LOGGING_STEPS=10
SAVE_STEPS=100

# LoRA parameters
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.0

# Other options
LOAD_IN_4BIT=True
USE_GRADIENT_CHECKPOINTING=True
SEED=42

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LLM Fine-tuning ===${NC}"
echo ""

# Check if training CSV exists
if [ ! -f "$TRAIN_CSV" ]; then
    echo -e "${RED}Error: Training CSV file not found: $TRAIN_CSV${NC}"
    echo -e "${YELLOW}Please update TRAIN_CSV in this script to point to your training data.${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Training CSV: ${YELLOW}${TRAIN_CSV}${NC}"
echo -e "  Base Model: ${YELLOW}${MODEL_NAME}${NC}"
echo -e "  Output Directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Max Sequence Length: ${YELLOW}${MAX_SEQ_LENGTH}${NC}"
echo ""
echo -e "${BLUE}Training Hyperparameters:${NC}"
echo -e "  Epochs: ${YELLOW}${NUM_TRAIN_EPOCHS}${NC}"
echo -e "  Batch Size: ${YELLOW}${PER_DEVICE_TRAIN_BATCH_SIZE}${NC}"
echo -e "  Gradient Accumulation Steps: ${YELLOW}${GRADIENT_ACCUMULATION_STEPS}${NC}"
echo -e "  Learning Rate: ${YELLOW}${LEARNING_RATE}${NC}"
echo -e "  LoRA r: ${YELLOW}${LORA_R}${NC}"
echo -e "  LoRA alpha: ${YELLOW}${LORA_ALPHA}${NC}"
echo ""

# Prepare command arguments
CMD_ARGS=(
    --train_csv "$TRAIN_CSV"
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --encoding "$ENCODING"
    --num-train-epochs "$NUM_TRAIN_EPOCHS"
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --seed "$SEED"
)

# Add optional arguments
if [ "$MAX_SAMPLES_PER_ROW" != "None" ]; then
    CMD_ARGS+=(--max-samples-per-row "$MAX_SAMPLES_PER_ROW")
fi

if [ "$MAX_TOTAL_SAMPLES" != "None" ]; then
    CMD_ARGS+=(--max-total-samples "$MAX_TOTAL_SAMPLES")
fi

if [ "$LOAD_IN_4BIT" = False ]; then
    CMD_ARGS+=(--no-4bit)
fi

if [ "$USE_GRADIENT_CHECKPOINTING" = False ]; then
    CMD_ARGS+=(--no-gradient-checkpointing)
fi

# Run fine-tuning
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Starting Fine-tuning...${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

if python "${PROJECT_ROOT}/src/llm_text_classification/finetune_llm.py" "${CMD_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Fine-tuning Complete!${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Model saved to: ${YELLOW}${OUTPUT_DIR}${NC}"
    echo -e "  - LoRA adapters: ${YELLOW}${OUTPUT_DIR}/${NC}"
    echo -e "  - Merged 16-bit model: ${YELLOW}${OUTPUT_DIR}/merged_16bit/${NC}"
    echo -e "  - Merged 4-bit model: ${YELLOW}${OUTPUT_DIR}/merged_4bit/${NC}"
    echo -e "  - Training info: ${YELLOW}${OUTPUT_DIR}/training_info.json${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  Fine-tuning Failed!${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 1
fi

