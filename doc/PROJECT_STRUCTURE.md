# KorEduBench Project Structure

This document describes the directory structure and organization of the KorEduBench project.

## Overview

```
KorEduBench/
├── src/                    # Source code directory
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── cosine_similarity/  # Cosine similarity evaluation
│   ├── cross_encoder/      # Cross-encoder training and evaluation
│   └── utils/              # Utility functions
├── dataset/                # Generated datasets
├── model/                  # Saved trained models
├── output/                 # Evaluation results and logs
│   ├── cosine_similarity/  # Cosine similarity results
│   └── cross_encoder/      # Cross-encoder results
├── scripts/                # Shell scripts for automation
└── doc/                    # Documentation
```

## Directory Details

### `src/` - Source Code

Contains all Python source code organized by functionality.

#### `src/preprocessing/`
Data preprocessing scripts that extract and prepare curriculum data.

- **`extract_standards.py`** - Extracts unique achievement standards from raw data
  - Input: Label directory with ZIP files containing JSON data
  - Output: `dataset/unique_achievement_standards.csv` (or with prefix like `training_`, `validation_`)
  
- **`add_text_to_standards.py`** - Adds text samples to each achievement standard
  - Input: Achievement standards CSV and label directory
  - Output: `dataset/text_achievement_standards.csv` with text columns added
  
- **`split_subject.py`** - Splits the dataset by subject for evaluation
  - Input: Text achievement standards CSV
  - Output: `dataset/subject_text{N}/` directory with per-subject CSV files

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
  - Output: `model/biencoder_finetuned/`

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

#### `src/utils/`
Utility functions and helper scripts.

- **`check_api_key_access.py`** - Checks API key access for external services

### `dataset/` - Generated Datasets

All preprocessed data files are stored here. The directory is created and populated by preprocessing scripts.

**Typical contents after preprocessing:**
```
dataset/
├── training_unique_achievement_standards.csv
├── training_text_achievement_standards.csv
├── training_subject_text20/
│   ├── 과학.csv
│   ├── 국어.csv
│   ├── 수학.csv
│   └── ...
├── validation_unique_achievement_standards.csv
├── validation_text_achievement_standards.csv
└── validation_subject_text20/
    ├── 과학.csv
    ├── 국어.csv
    └── ...
```

**File naming convention:**
- `{prefix}_unique_achievement_standards.csv` - Unique achievement standards
- `{prefix}_text_achievement_standards.csv` - Standards with text samples
- `{prefix}_subject_text{N}/` - Per-subject CSV files (N = max text samples per row)

### `model/` - Trained Models

Stores fine-tuned models generated during training.

**Contents:**
```
model/
├── biencoder_finetuned/    # Fine-tuned dual encoder (from train_dual_encoder.py)
└── cross_finetuned/        # Fine-tuned cross encoder (from finetune_cross_encoder.py)
```

Each model directory contains:
- Model weights
- Configuration files
- Tokenizer files

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
    "folder": "training",
    "model_name": "jhgan/ko-sroberta-multitask",
    "subject": "과학",
    "num_standards": 150,
    "max_samples_per_row": 20,
    "total_samples": 3000,
    "top1_acc": 0.7234,
    "top3_acc": 0.8456,
    "top10_acc": 0.9123,
    "top20_acc": 0.9456,
    "top40_acc": 0.9678,
    "top60_acc": 0.9789,
    "mrr": 0.7891
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
    "folder": "validation",
    "bi_model": "jhgan/ko-sroberta-multitask",
    "cross_model": "model/cross_finetuned",
    "subject": "과학",
    "num_standards": 150,
    "max_samples_per_row": 20,
    "total_samples": 3000,
    "top_k": 20,
    "top1_acc": 0.8234,
    "top3_acc": 0.9156,
    "top10_acc": 0.9623,
    "top20_acc": 0.9756,
    "mrr": 0.8591
  },
  ...
]
```

**`logs/{subject}_wrongs.txt`** - Contains up to 100 randomly sampled misclassified examples with:
- Input text
- True achievement standard code and content
- Predicted achievement standard code and content

### `scripts/` - Automation Scripts

Shell scripts for running the entire pipeline.

- **`preprocess.sh`** - Runs the full preprocessing pipeline
  - Extracts standards
  - Adds text samples
  - Splits by subject
  - Processes both Training and Validation datasets
  
- **`cosine_similarity.sh`** - Runs cosine similarity evaluation
  - Evaluates on training dataset
  - Saves results to `output/cosine_similarity/results.json`
  
- **`cross_encoder.sh`** - Trains and evaluates cross-encoder
  - Fine-tunes cross-encoder on training data
  - Evaluates on validation data
  - Saves model and results

**Usage:**
```bash
cd scripts

# Step 1: Preprocess data
bash preprocess.sh

# Step 2: Run cosine similarity baseline
bash cosine_similarity.sh

# Step 3: Train and evaluate cross-encoder
bash cross_encoder.sh
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
dataset/{prefix}_unique_achievement_standards.csv
    ↓
[add_text_to_standards.py]
    ↓
dataset/{prefix}_text_achievement_standards.csv
    ↓
[split_subject.py]
    ↓
dataset/{prefix}_subject_text{N}/{subject}.csv
    ↓
├─→ [eval_cosine_similarity.py] → output/cosine_similarity/results.json
│
├─→ [train_dual_encoder.py] → model/biencoder_finetuned/
│
└─→ [finetune_cross_encoder.py] → model/cross_finetuned/
        ↓
    [eval_cross_encoder.py] → output/cross_encoder/results_rerank.json
                             → output/cross_encoder/logs/{subject}_wrongs.txt
```

## Key Features

1. **Organized Structure**: Clear separation between source code, data, models, and outputs
2. **Automatic Path Resolution**: Scripts automatically find project root
3. **Consistent Naming**: Prefixes (training/validation) and suffixes maintain consistency
4. **Reproducible Pipeline**: Shell scripts automate the entire workflow
5. **Version Control Friendly**: Generated files (dataset/, model/, output/) can be gitignored

## Notes

- The `dataset/`, `model/`, and `output/` directories are created automatically by scripts
- All paths are relative to the project root for portability
- Shell scripts use `PROJECT_ROOT` variable for consistency
- JSON result files use append logic to accumulate results from multiple runs

