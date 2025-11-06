#!/bin/bash
# Advanced Bi-Encoder Training Script

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters (can be overridden)
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-model/biencoder_advanced}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"

echo "======================================================================"
echo "Advanced Bi-Encoder Training"
echo "======================================================================"
echo "Input CSV: $INPUT_CSV"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "======================================================================"
echo ""

python src/cosine_similarity/train_advanced_biencoder.py \
    --input_csv "$INPUT_CSV" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --hard_negative_mining \
    --hard_negative_epochs 2 4 6 \
    --mixed_precision \
    --early_stopping_patience 3

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "======================================================================"

