#!/bin/bash
# Train with larger model and more advanced techniques for better performance

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use larger model and label smoothing for better generalization
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:-model/achievement_classifier_large}"
BATCH_SIZE="${BATCH_SIZE:-24}"  # Slightly smaller for large model
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-1e-5}"  # Lower LR for large model
MAX_LENGTH="${MAX_LENGTH:-384}"  # Longer sequences
LOSS_TYPE="${LOSS_TYPE:-label_smoothing}"

echo "======================================================================"
echo "Advanced Multi-Class Classifier Training (Large Model)"
echo "======================================================================"
echo "Input CSV: $INPUT_CSV"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUM"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUM))"
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
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --loss_type "$LOSS_TYPE" \
    --label_smoothing 0.1 \
    --mixed_precision \
    --early_stopping_patience 4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --dropout 0.15 \
    --pooling mean

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/best_model"
echo "======================================================================"

