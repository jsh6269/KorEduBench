#!/bin/bash
# Advanced Training with Larger Model (더 큰 모델로 더 나은 성능)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 더 큰 배치 사이즈와 더 나은 모델 사용
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-large}"  # Large model for better performance
OUTPUT_DIR="${OUTPUT_DIR:-model/biencoder_advanced_large}"
BATCH_SIZE="${BATCH_SIZE:-8}"  # Smaller batch size for large model
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"  # Larger accumulation
EPOCHS="${EPOCHS:-12}"
LR="${LR:-1e-5}"  # Lower learning rate for large model

echo "======================================================================"
echo "Advanced Bi-Encoder Training (Large Model)"
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
    --hard_negative_epochs 2 4 6 8 \
    --mixed_precision \
    --early_stopping_patience 4

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "======================================================================"

