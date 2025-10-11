#!/bin/bash

# Naive Cosine Similarity Evaluation
# This script runs batch cosine similarity evaluation on subject CSV files

# Set paths
DATASET_FOLDER="../dataset/training_subject_text20"

# Check if folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder not found: $DATASET_FOLDER"
    exit 1
fi

# Change to cosine_similarity directory
cd ../cosine_similarity || exit 1

echo "Running batch cosine similarity evaluation..."
echo "Dataset folder: $DATASET_FOLDER"

# Run batch cosine similarity
python batch_cosine_similarity.py --folder_path "$DATASET_FOLDER"

echo "Cosine similarity evaluation completed!"

