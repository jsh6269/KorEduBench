#!/bin/bash

# Cross Encoder Training and Evaluation
# This script fine-tunes and evaluates a cross encoder model

# Set paths
TRAIN_CSV="../dataset/training_subject_text20/과학.csv"
VALIDATION_CSV="../dataset/validation_subject_text20/과학.csv"
MODEL_SAVE_PATH="./cross_finetuned"

# Check if input CSVs exist
if [ ! -f "$TRAIN_CSV" ]; then
    echo "Error: Training CSV not found: $TRAIN_CSV"
    echo "Please provide a valid CSV file path"
    exit 1
fi

if [ ! -f "$VALIDATION_CSV" ]; then
    echo "Error: Validation CSV not found: $VALIDATION_CSV"
    echo "Please provide a valid CSV file path"
    exit 1
fi

# Change to cross_encoder directory (assuming script is in scripts/ directory)
cd ../cross_encoder || exit 1

echo "Starting Cross Encoder training and evaluation..."
echo "Training CSV: $TRAIN_CSV"
echo "Validation CSV: $VALIDATION_CSV"
echo "Model save path: $MODEL_SAVE_PATH"
echo ""

# Step 1: Fine-tune cross encoder
echo "Step 1: Fine-tuning cross encoder..."
python finetune_cross_encoder.py --input_csv "$TRAIN_CSV"

if [ $? -ne 0 ]; then
    echo "Error: Fine-tuning failed"
    exit 1
fi

echo "Fine-tuning completed!"
echo ""

# Step 2: Evaluate cross encoder
echo "Step 2: Evaluating cross encoder..."
python eval_cross_encoder.py --input_csv "$VALIDATION_CSV"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "Cross encoder training and evaluation completed!"

