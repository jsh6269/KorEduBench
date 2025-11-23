# KorEduBench Project Structure

This document describes the directory structure and organization of the KorEduBench project.

## Overview

```
KorEduBench/
├── src/                         # Source code directory
│   ├── preprocessing/           # Data preprocessing scripts
│   ├── cosine_similarity/       # Cosine similarity evaluation
│   ├── cross_encoder/           # Cross-encoder training and evaluation
│   ├── classification/          # Multi-class classifier training
│   ├── llm_text_classification/ # LLM-based text classification
│   ├── test/                    # Test scripts
│   └── utils/                   # Utility functions
├── dataset/                     # Generated datasets
├── model/                       # Saved trained models
├── output/                      # Evaluation results and logs
│   ├── cosine_similarity/       # Cosine similarity results
│   ├── cross_encoder/           # Cross-encoder results
│   └── llm_text_classification/ # LLM classification results
├── scripts/                     # Shell scripts for automation
└── doc/                         # Documentation
```

## Directory Details

### `src/` - Source Code

Contains all Python source code organized by functionality.

#### `src/preprocessing/`
Data preprocessing scripts that extract and prepare curriculum data.

- **`extract_standards.py`** - Extracts unique achievement standards from raw data
  - Input: Label directory with ZIP files containing JSON data
  - Output: `dataset/unique_achievement_standards.csv`
  
- **`add_text_to_standards.py`** - Adds text samples to each achievement standard
  - Input: Achievement standards CSV and label directory
  - Output: `dataset/text_achievement_standards.csv` with text columns added
  
- **`split_subject.py`** - Splits the dataset by subject and creates train/validation splits
  - Input: Text achievement standards CSV
  - Output: 
    - `dataset/train.csv`, `dataset/valid.csv` - Complete train/validation datasets
    - `dataset/train_80/`, `dataset/valid_80/` - Per-subject CSV files (80 texts per standard)
    - `dataset/few_shot_examples/` - Few-shot example JSON files for each subject
    - `dataset/insufficient_text.csv` - Standards with insufficient text samples

#### `src/cosine_similarity/`
Cosine similarity-based curriculum mapping evaluation.

- **`eval_cosine_similarity.py`** - Evaluates cosine similarity baseline on a single CSV
  - Uses SentenceTransformer models for embedding
  - Computes top-k accuracy and MRR (Mean Reciprocal Rank)
  - Output: `output/cosine_similarity/results.json`
  
- **`batch_cosine_similarity.py`** - Batch evaluation on multiple CSV files in a folder
  - Iterates through all CSV files in a directory
  - Appends results to shared JSON file
  
- **`train_dual_encoder.py`** - Fine-tunes a dual (bi-encoder) model
  - Trains on positive and negative text-standard pairs
  - Output: Trained bi-encoder model
  
- **`train_advanced_biencoder.py`** - Advanced bi-encoder training
  - Enhanced training with advanced techniques
  - Output: Advanced bi-encoder model

#### `src/cross_encoder/`
Cross-encoder reranking for improved accuracy.

- **`finetune_cross_encoder.py`** - Fine-tunes a cross-encoder model
  - Trains on text-standard pairs with binary labels
  - Output: `model/cross_finetuned/`
  
- **`eval_cross_encoder.py`** - Evaluates bi-encoder + cross-encoder pipeline
  - First retrieves top-k candidates with bi-encoder
  - Then reranks with cross-encoder
  - Output: `output/cross_encoder/results_rerank.json`
  - Logs: `output/cross_encoder/logs/` (contains misclassified samples)

#### `src/classification/`
Traditional multi-class classifier training and prediction.

- **`train_multiclass_classifier.py`** - Trains a multi-class classifier on achievement standards
  - Uses transformer-based models for classification
  - Output: Trained classifier model
  
- **`predict_multiclass.py`** - Predicts achievement standards using trained classifier
  - Evaluates classification performance
  - Output: Prediction results and metrics

#### `src/llm_text_classification/`
LLM (Large Language Model) based text classification evaluation.

- **`finetune_llm.py`** - Fine-tunes LLM for achievement standard classification
  - Trains generative models on curriculum mapping task
  - Output: Fine-tuned LLM model
  
- **`eval_llm.py`** - Evaluates pre-trained LLM on classification task
  - Uses zero-shot or few-shot prompting
  - Output: `output/llm_text_classification/results.json`
  - Logs: `output/llm_text_classification/logs/` (misclassified samples)
  
- **`eval_finetune_llm.py`** - Evaluates fine-tuned LLM on classification task
  - Tests fine-tuned model performance
  - Output: `output/llm_text_classification/finetuned_results.json`
  - Logs: `output/llm_text_classification/finetuned_logs/` (correct/wrong samples)

#### `src/test/`
Test scripts for development and validation.

#### `src/utils/`
Utility functions and helper scripts.

- **`check_api_key_access.py`** - Checks API key access for external services

### `dataset/` - Generated Datasets

All preprocessed data files are stored here. The directory is created and populated by preprocessing scripts.

**Typical contents after preprocessing:**
```
dataset/
├── unique_achievement_standards.csv      # All unique achievement standards
├── text_achievement_standards.csv        # Standards with text samples
├── train.csv                             # Training dataset (all subjects)
├── valid.csv                             # Validation dataset (all subjects)
├── train_80/                             # Training data by subject (80 texts per standard)
│   ├── 과학.csv
│   ├── 국어.csv
│   ├── 수학.csv
│   ├── 영어.csv
│   ├── 사회.csv
│   ├── 사회문화.csv
│   ├── 도덕.csv
│   ├── 기술가정.csv
│   └── 정보.csv
├── valid_80/                             # Validation data by subject (80 texts per standard)
│   ├── 과학.csv
│   ├── 국어.csv
│   └── ...
├── few_shot_examples/                    # Few-shot examples for LLM prompting
│   ├── 과학.json
│   ├── 국어.json
│   └── ...
└── insufficient_text.csv                 # Standards with insufficient text samples
```

**File descriptions:**
- `unique_achievement_standards.csv` - Deduplicated achievement standards from raw data
- `text_achievement_standards.csv` - Standards with associated text samples
- `train.csv` / `valid.csv` - Complete training/validation datasets
- `train_80/` / `valid_80/` - Per-subject CSV files (80 text samples per achievement standard)
- `few_shot_examples/` - JSON files containing few-shot examples for each subject (for LLM prompting)
- `insufficient_text.csv` - Achievement standards that don't have enough text samples

### `model/` - Trained Models

Stores fine-tuned models and model outputs generated during training.

**Contents:**
```
model/
├── cross_encoder/          # Cross-encoder models (training checkpoints)
├── cross_finetuned/        # Fine-tuned cross-encoder (from finetune_cross_encoder.py)
├── finetuned_llm/          # Fine-tuned LLM (from finetune_llm.py)
├── few-shot/               # Few-shot LLM evaluation outputs
└── zero-shot/              # Zero-shot LLM evaluation outputs
```

Each model directory contains:
- Model weights (for trained models)
- Configuration files
- Tokenizer files
- Evaluation outputs (for LLM directories)

### `output/` - Evaluation Results

All evaluation results and logs are stored here.

#### `output/cosine_similarity/`
```
output/cosine_similarity/
└── results.json            # Cosine similarity evaluation results
```

**`results.json` format:**
```json
[
  {
    "folder": "valid_80",
    "model_name": "jhgan/ko-sroberta-multitask",
    "subject": "과학",
    "num_standards": 190,
    "max_samples_per_row": 80,
    "total_samples": 15200,
    "top1_acc": 0.4241,
    "top3_acc": 0.6135,
    "top10_acc": 0.7741,
    "top20_acc": 0.8431,
    "top40_acc": 0.8989,
    "mrr": 0.5447
  },
  ...
]
```

#### `output/cross_encoder/`
```
output/cross_encoder/
├── results_rerank.json     # Cross-encoder reranking results
└── logs/                   # Misclassified sample logs
    ├── 과학_wrongs.txt
    └── ...
```

**`results_rerank.json` format:**
```json
[
  {
    "folder": "valid_80",
    "bi_model": "jhgan/ko-sroberta-multitask",
    "cross_model": "../../model/cross_finetuned",
    "subject": "과학",
    "num_standards": 190,
    "max_samples_per_row": 80,
    "total_samples": 15200,
    "top_k": 20,
    "top1_acc": 0.4849,
    "top3_acc": 0.6945,
    "top10_acc": 0.818,
    "top20_acc": 0.8431,
    "mrr": 0.603
  },
  ...
]
```

**`logs/{subject}_wrongs.txt`** - Contains up to 100 randomly sampled misclassified examples with:
- Input text
- True achievement standard code and content
- Predicted achievement standard code and content

#### `output/llm_text_classification/`
```
output/llm_text_classification/
├── results.json              # Pre-trained LLM evaluation results
├── finetuned_results.json    # Fine-tuned LLM evaluation results
├── logs/                     # Pre-trained LLM logs
│   ├── {subject}_corrects.txt
│   └── {subject}_wrongs.txt
└── finetuned_logs/           # Fine-tuned LLM logs
    ├── {subject}_corrects.txt
    └── {subject}_wrongs.txt
```

**`results.json` / `finetuned_results.json` format:**
```json
[
  {
    "folder": "valid_80",
    "model_path": "/path/to/model",
    "base_model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "subject": "과학",
    "num_standards": 190,
    "num_candidates": 120,
    "max_candidates": 120,
    "max_samples_per_row": 80,
    "total_samples": 100,
    "correct": 75,
    "accuracy": 0.75,
    "mrr": 0.82,
    "exact_match_count": 68,
    "exact_match_percentage": 0.68,
    "match_type_distribution": {
      "exact": 68.0,
      "partial": 32.0
    },
    "max_new_tokens": 50,
    "temperature": 0.1,
    "max_input_length": 6144,
    "truncated_count": 0,
    "truncated_percentage": 0.0,
    "training_info": {}
  },
  ...
]
```

**`logs/{subject}_corrects.txt`** and **`logs/{subject}_wrongs.txt`** - Contains correctly/incorrectly classified examples with:
- Input text
- True achievement standard code and content
- Predicted achievement standard code and content
- Model reasoning/explanation

### `scripts/` - Automation Scripts

Shell scripts for running the entire pipeline.

#### Data Preprocessing
- **`preprocess.sh`** - Runs the full preprocessing pipeline
  - Extracts standards
  - Adds text samples
  - Splits by subject and creates train/validation splits
  - Generates few-shot examples for LLM evaluation

#### Embedding-based Approaches
- **`cosine_similarity.sh`** - Runs cosine similarity evaluation
  - Evaluates on validation dataset
  - Saves results to `output/cosine_similarity/results.json`
  
- **`cross_encoder.sh`** - Trains and evaluates cross-encoder
  - Fine-tunes cross-encoder on training data
  - Evaluates on validation data with reranking
  - Saves model and results

#### Multi-class Classification
- **`train_classifier.sh`** - Trains standard multi-class classifier
  - Basic training configuration
  
- **`train_classifier_focal.sh`** - Trains classifier with focal loss
  - Handles class imbalance with focal loss
  
- **`train_classifier_large.sh`** - Trains large-scale classifier
  - Uses larger model variants
  
- **`train_advanced.sh`** - Trains advanced classifier
  - Advanced training configuration
  
- **`train_advanced_large.sh`** - Trains large advanced classifier
  - Combines advanced techniques with large models

#### LLM-based Text Classification
- **`llm_text_classification.sh`** - Evaluates pre-trained LLM
  - Zero-shot or few-shot evaluation
  - Saves results to `output/llm_text_classification/results.json`
  
- **`finetuning_llm.sh`** - Fine-tunes LLM for classification
  - Trains generative model on curriculum mapping
  
- **`finetune_llm_text_classification.sh`** - Evaluates fine-tuned LLM
  - Tests fine-tuned model performance
  - Saves results to `output/llm_text_classification/finetuned_results.json`

#### Other
- **`checkpoints/`** - Stores training checkpoint information

**Basic Usage:**
```bash
cd scripts

# Step 1: Preprocess data
bash preprocess.sh

# Step 2: Run cosine similarity baseline
bash cosine_similarity.sh

# Step 3: Train and evaluate cross-encoder
bash cross_encoder.sh

# Step 4: Evaluate LLM
bash llm_text_classification.sh

# Step 5: Fine-tune and evaluate LLM
bash finetuning_llm.sh
bash finetune_llm_text_classification.sh
```

### `doc/` - Documentation

Project documentation files.

- **`PROJECT_STRUCTURE.md`** (this file) - Project directory structure
- **`CODE_ANALYSIS.md`** - Code analysis and technical details

## Path Resolution

All Python scripts use **absolute path resolution** based on the project root:

```python
from pathlib import Path

# Get project root (3 levels up from src/{module}/{script}.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
```

This ensures that:
- Scripts can be run from any directory
- All outputs go to the correct location relative to project root
- No issues with relative path confusion

## Data Flow

```
Raw Data (ZIP files with JSON)
    ↓
[extract_standards.py]
    ↓
dataset/unique_achievement_standards.csv
    ↓
[add_text_to_standards.py]
    ↓
dataset/text_achievement_standards.csv
    ↓
[split_subject.py]
    ↓
dataset/train.csv, dataset/valid.csv
dataset/train_80/{subject}.csv, dataset/valid_80/{subject}.csv
dataset/few_shot_examples/{subject}.json
    ↓
├─→ [eval_cosine_similarity.py] → output/cosine_similarity/results.json
│
├─→ [finetune_cross_encoder.py] → model/cross_finetuned/
│       ↓
│   [eval_cross_encoder.py] → output/cross_encoder/results_rerank.json
│                            → output/cross_encoder/logs/{subject}_wrongs.txt
│
├─→ [train_multiclass_classifier.py] → model/classifier/
│       ↓
│   [predict_multiclass.py] → prediction results
│
└─→ [finetune_llm.py] → model/llm_finetuned/
        ↓
    ├─→ [eval_llm.py] → output/llm_text_classification/results.json
    │                  → output/llm_text_classification/logs/
    │
    └─→ [eval_finetune_llm.py] → output/llm_text_classification/finetuned_results.json
                                 → output/llm_text_classification/finetuned_logs/
```

## Key Features

1. **Organized Structure**: Clear separation between source code, data, models, and outputs
2. **Automatic Path Resolution**: Scripts automatically find project root
3. **Consistent Naming**: Clear train/valid split with standardized naming conventions
4. **Reproducible Pipeline**: Shell scripts automate the entire workflow
5. **Version Control Friendly**: Generated files (dataset/, model/, output/) can be gitignored

## Notes

- The `dataset/`, `model/`, and `output/` directories are created automatically by scripts
- All paths are relative to the project root for portability
- Shell scripts use `PROJECT_ROOT` variable for consistency
- JSON result files use append logic to accumulate results from multiple runs
- Dataset uses fixed 80 text samples per achievement standard in train_80/ and valid_80/
- Few-shot examples are automatically generated for LLM evaluation
- The project now includes multiple evaluation approaches: embedding-based, classification-based, and LLM-based

