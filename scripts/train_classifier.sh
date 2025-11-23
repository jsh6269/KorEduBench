#!/bin/bash
# Train Multi-Class Achievement Standard Classifier

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
INPUT_CSV="${INPUT_CSV:-dataset/train.csv}"
BASE_MODEL="${BASE_MODEL:-klue/roberta-large}"
OUTPUT_DIR="${OUTPUT_DIR:-model/achievement_classifier}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LOSS_TYPE="${LOSS_TYPE:-ce}"
NO_EVAL="${NO_EVAL:-true}"
RESUME_FROM="${RESUME_FROM:-model/achievement_classifier/checkpoint_epoch_3}"

echo "======================================================================"
echo "Multi-Class Achievement Standard Classifier Training"
echo "======================================================================"
echo "Input CSV: $INPUT_CSV"
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Max Length: $MAX_LENGTH"
echo "Loss Type: $LOSS_TYPE"
echo "No Evaluation: $NO_EVAL"
echo "======================================================================"
echo ""

python src/classification/train_multiclass_classifier.py \
    --input_csv "$INPUT_CSV" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --loss_type "$LOSS_TYPE" \
    --mixed_precision \
    --early_stopping_patience 3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --pooling cls \
    --no_eval "$NO_EVAL" \
    --resume_from "$RESUME_FROM"

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/best_model"
echo "======================================================================"

