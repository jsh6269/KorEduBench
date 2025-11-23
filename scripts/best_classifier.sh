#!/bin/bash
# Evaluate Multi-Class Achievement Standard Classifier
# Evaluate multiple checkpoints and select the best model

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
OUTPUT_DIR="${OUTPUT_DIR:-model/achievement_classifier}"
INPUT_CSV="${INPUT_CSV:-dataset/valid.csv}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-8 9 10}"
METRIC_FOR_SELECTION="${METRIC_FOR_SELECTION:-f1_weighted}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo "======================================================================"
echo "Multi-Class Achievement Standard Classifier - Evaluation"
echo "======================================================================"
echo "Output Directory: $OUTPUT_DIR"
echo "Input CSV: $INPUT_CSV"
echo "Checkpoint Epochs: $CHECKPOINT_EPOCHS"
echo "Metric for Selection: $METRIC_FOR_SELECTION"
echo "Batch Size: $BATCH_SIZE"
echo "======================================================================"
echo ""

python src/classification/eval_multiclass_classifier.py \
    --output_dir "$OUTPUT_DIR" \
    --input_csv "$INPUT_CSV" \
    --checkpoint_epochs $CHECKPOINT_EPOCHS \
    --metric_for_selection "$METRIC_FOR_SELECTION" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "Best model saved to: $OUTPUT_DIR/best_model"
echo "Evaluation summary: $OUTPUT_DIR/checkpoint_evaluation_summary.json"
echo "======================================================================"

