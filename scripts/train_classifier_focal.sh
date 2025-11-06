#!/bin/bash
# Train with Focal Loss for handling class imbalance

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Focal loss is good when some classes have much fewer samples than others
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:-model/achievement_classifier_focal}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-12}"
LR="${LR:-2e-5}"
LOSS_TYPE="${LOSS_TYPE:-focal}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

echo "======================================================================"
echo "Multi-Class Classifier Training with Focal Loss"
echo "======================================================================"
echo "Input CSV: $INPUT_CSV"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Loss: Focal Loss (gamma=$FOCAL_GAMMA)"
echo "======================================================================"
echo ""

python src/classification/train_multiclass_classifier.py \
    --input_csv "$INPUT_CSV" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --loss_type "$LOSS_TYPE" \
    --focal_gamma "$FOCAL_GAMMA" \
    --focal_alpha 1.0 \
    --mixed_precision \
    --early_stopping_patience 3

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "======================================================================"

