#!/bin/bash
# Train Multi-Class Achievement Standard Classifier

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:-model/achievement_classifier}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LOSS_TYPE="${LOSS_TYPE:-ce}"

echo "======================================================================"
echo "Multi-Class Achievement Standard Classifier Training"
echo "======================================================================"
echo "Input CSV: $INPUT_CSV"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Max Length: $MAX_LENGTH"
echo "Loss Type: $LOSS_TYPE"
echo "======================================================================"
echo ""

python src/classification/train_multiclass_classifier.py \
    --input_csv "$INPUT_CSV" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --loss_type "$LOSS_TYPE" \
    --mixed_precision \
    --early_stopping_patience 3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --pooling cls

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/best_model"
echo "======================================================================"

