#!/bin/bash
# Evaluate Best Model using predict_multiclass.py
# This script evaluates the best_model using predict_multiclass.py which provides
# detailed top-k accuracy metrics and saves results to output/classification/results.json

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
MODEL_DIR="${MODEL_DIR:-model/achievement_classifier/best_model}"
INPUT_CSV="${INPUT_CSV:-dataset/valid.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
TOP_K="${TOP_K:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ENCODING="${ENCODING:-utf-8}"

echo "======================================================================"
echo "Best Model Evaluation using predict_multiclass.py"
echo "======================================================================"
echo "Model Directory: $MODEL_DIR"
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: ${OUTPUT_CSV:-Not specified (will not save predictions)}"
echo "Top-K: $TOP_K"
echo "Batch Size: $BATCH_SIZE"
echo "Encoding: $ENCODING"
echo "======================================================================"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo "Please make sure the best model exists or set MODEL_DIR environment variable."
    exit 1
fi

# Check if input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input CSV not found: $INPUT_CSV"
    echo "Please provide a valid CSV file or set INPUT_CSV environment variable."
    exit 1
fi

# Build command
CMD="python src/classification/predict_multiclass.py \
    --model_dir \"$MODEL_DIR\" \
    --input_csv \"$INPUT_CSV\" \
    --top_k $TOP_K \
    --batch_size $BATCH_SIZE \
    --encoding $ENCODING"

# Add output_csv if specified
if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --output_csv \"$OUTPUT_CSV\""
fi

# Execute command
eval $CMD

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: output/classification/results.json"
if [ -n "$OUTPUT_CSV" ]; then
    echo "Predictions saved to: $OUTPUT_CSV"
fi
echo "======================================================================"

