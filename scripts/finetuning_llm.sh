#!/bin/bash

# LLM Fine-tuning Script
# This script fine-tunes an LLM on a training dataset and saves the model

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Training configuration
TRAIN_DIR="${PROJECT_ROOT}/dataset/train_80"  # Directory containing training CSV files
MODEL_NAME="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_llm"
MAX_SEQ_LENGTH=3000
MAX_SAMPLES_PER_ROW=20     # Train with n sample per achievement standard
MAX_TOTAL_SAMPLES=None    # No limit on total samples (after per-row filtering)
MAX_CANDIDATES=30         # Limit candidates per prompt to avoid overly long prompts
ENCODING="utf-8"

# Training hyperparameters
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
WARMUP_STEPS=5
LOGGING_STEPS=20
SAVE_STEPS=50

# LoRA parameters
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.0

# Other options
LOAD_IN_4BIT=True
SEED=42

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LLM Fine-tuning ===${NC}"
echo ""

# Check if training directory exists
if [ ! -d "$TRAIN_DIR" ]; then
    echo -e "${RED}Error: Training directory not found: $TRAIN_DIR${NC}"
    echo -e "${YELLOW}Please update TRAIN_DIR in this script to point to your training data.${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Training Directory: ${YELLOW}${TRAIN_DIR}${NC}"
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
    --train_dir "$TRAIN_DIR"
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

if [ "$MAX_CANDIDATES" != "None" ]; then
    CMD_ARGS+=(--max-candidates "$MAX_CANDIDATES")
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
