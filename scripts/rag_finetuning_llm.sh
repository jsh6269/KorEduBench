#!/bin/bash

# RAG LLM Fine-tuning Script
# This script fine-tunes an LLM on a training dataset using RAG workflow and saves the model

# set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Training configuration
TRAIN_DIR="${PROJECT_ROOT}/dataset/train_80"  # Directory containing training CSV files (each CSV file is used as train_csv for infer_top_k)
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"  # Path to model directory for infer_top_k
MODEL_NAME="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_rag_llm"
MAX_SEQ_LENGTH=2600
NUM_SAMPLES=2500          # Target number of samples per CSV file (default: None, use all)
ENCODING="utf-8"

# RAG parameters
TOP_K=20                   # Number of top candidates to retrieve
INFER_DEVICE="cuda"        # Device for infer_top_k execution
NUM_EXAMPLES_FEW_SHOT=5    # Number of few-shot examples (0 to disable few-shot)

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

echo -e "${GREEN}=== RAG LLM Fine-tuning ===${NC}"
echo ""

# Check if training directory exists
if [ ! -d "$TRAIN_DIR" ]; then
    echo -e "${RED}Error: Training directory not found: $TRAIN_DIR${NC}"
    echo -e "${YELLOW}Please update TRAIN_DIR in this script to point to your training data.${NC}"
    exit 1
fi


# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_DIR${NC}"
    echo -e "${YELLOW}Please update MODEL_DIR in this script to point to your classification model.${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Training Directory: ${YELLOW}${TRAIN_DIR}${NC}"
echo -e "  Model Directory (for RAG): ${YELLOW}${MODEL_DIR}${NC}"
echo -e "  Base Model: ${YELLOW}${MODEL_NAME}${NC}"
echo -e "  Output Directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Max Sequence Length: ${YELLOW}${MAX_SEQ_LENGTH}${NC}"
echo ""
echo -e "${BLUE}RAG Parameters:${NC}"
echo -e "  Top-k: ${YELLOW}${TOP_K}${NC}"
echo -e "  Infer Device: ${YELLOW}${INFER_DEVICE}${NC}"
echo -e "  Few-shot Examples: ${YELLOW}${NUM_EXAMPLES_FEW_SHOT}${NC}"
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
    --model-dir "$MODEL_DIR"
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --top-k "$TOP_K"
    --infer-device "$INFER_DEVICE"
    --num-examples-few-shot "$NUM_EXAMPLES_FEW_SHOT"
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
if [ "$NUM_SAMPLES" != "None" ]; then
    CMD_ARGS+=(--num-samples "$NUM_SAMPLES")
fi

if [ "$LOAD_IN_4BIT" = False ]; then
    CMD_ARGS+=(--no-4bit)
fi

# Run fine-tuning
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Starting Fine-tuning...${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

if python "${PROJECT_ROOT}/src/rag_llm_text_classification/rag_finetune_llm.py" "${CMD_ARGS[@]}"; then
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
