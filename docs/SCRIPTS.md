# Shell Scripts Documentation

This document describes the execution methods and usage of shell scripts located in the `scripts/` directory.

## Table of Contents

### Data Preprocessing

1. [preprocess.sh](#1-preprocesssh) - Data preprocessing pipeline

### Embedding-based Approaches

2. [cosine_similarity.sh](#2-cosine_similaritysh) - Cosine similarity evaluation
3. [cross_encoder.sh](#3-cross_encodersh) - Cross-encoder training and evaluation

### Multi-class Classification

4. [train_classifier.sh](#4-train_classifiersh) - Multi-class classifier training
5. [train_advanced.sh](#5-train_advancedsh) - Advanced multi-class classifier training
6. [eval_classifier.sh](#6-eval_classifiersh) - Multi-class classifier evaluation

### LLM-based Text Classification

7. [llm_text_classification.sh](#7-llm_text_classificationsh) - LLM evaluation
8. [finetuning_llm.sh](#8-finetuning_llmsh) - LLM fine-tuning
9. [finetune_llm_text_classification.sh](#9-finetune_llm_text_classificationsh) - Fine-tuned LLM evaluation

### RAG-based LLM Text Classification

10. [rag_llm_text_classification.sh](#10-rag_llm_text_classificationsh) - RAG LLM evaluation
11. [api_rag_llm_text_classification.sh](#11-api_rag_llm_text_classificationsh) - API-based RAG LLM evaluation
12. [rag_finetuning_llm.sh](#12-rag_finetuning_llmsh) - RAG LLM fine-tuning
13. [rag_finetune_llm_text_classification.sh](#13-rag_finetune_llm_text_classificationsh) - Fine-tuned RAG LLM evaluation

---

## 1. preprocess.sh

### Overview

Preprocessing script for the AI Hub curriculum-level subject-specific dataset. Performs achievement standard extraction, text sample addition and train/validation splitting.  
You can also proceed using well-organized sample dataset archived at `dataset/dataset_bundle.tar.gz`.

### Prerequisites

- Download the [Curriculum-level Subject-specific Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71855)
- Verify the path to the `label` directory in the dataset

### Configuration

Modify the following variables within the script:

```bash
BASE_DIR="/mnt/e/2025_2_KorEduBench"  # Base path to the dataset
MAX_TEXTS=80                           # Maximum number of text samples per achievement standard
```

### Execution

```bash
cd scripts
bash preprocess.sh
```

### Processing Pipeline

1. **Step 1: Achievement Standard Extraction**

   - Executes `extract_standards.py`
   - Input: `{BASE_DIR}/label/`
   - Output: `dataset/unique_achievement_standards.csv`

2. **Step 2: Text Sample Addition**

   - Executes `add_text_to_standards.py`
   - Adds up to `MAX_TEXTS` text samples per achievement standard
   - Output: `dataset/text_achievement_standards.csv`

3. **Step 3: Train/Validation Split and Subject-wise Partitioning**
   - Executes `split_subject.py`
   - Performs train/validation split (80/20)
   - Generates subject-specific CSV files
   - Generates few-shot example JSON files
   - Outputs:
     - `dataset/train.csv`, `dataset/valid.csv`
     - `dataset/train_80/{subject}.csv`, `dataset/valid_80/{subject}.csv`
     - `dataset/few_shot_examples/{subject}.json`
     - `dataset/insufficient_text.csv`

### Output Structure

```
dataset/
├── unique_achievement_standards.csv
├── text_achievement_standards.csv
├── train.csv                  # Complete training data
├── valid.csv                  # Complete validation data
├── train_80/                  # Subject-specific training data (80 texts/standard)
│   ├── 과학.csv
│   ├── 국어.csv
│   ├── 수학.csv
│   ├── 영어.csv
│   ├── 사회.csv
│   ├── 사회문화.csv
│   ├── 도덕.csv
│   ├── 기술가정.csv
│   └── 정보.csv
├── valid_80/                  # Subject-specific validation data
│   └── ...
├── few_shot_examples/         # LLM few-shot examples
│   ├── 과학.json
│   ├── 국어.json
│   └── ...
└── insufficient_text.csv      # Achievement standards with insufficient text samples
```

---

## 2. cosine_similarity.sh

### Overview

Performs cosine similarity-based evaluation on subject-specific CSV files using the `jhgan/ko-sroberta-multitask` model.

### Prerequisites

- `preprocess.sh` execution completed
- `dataset/valid_80/` directory exists

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"  # Dataset folder for evaluation
```

### Execution

```bash
cd scripts
bash cosine_similarity.sh
```

### Methodology

1. Discovers all CSV files in the `valid_80` folder (9 subjects)
2. Performs cosine similarity evaluation for each CSV file
3. Embeds texts and achievement standards using a SentenceTransformer model
4. Computes top-k accuracy and MRR (Mean Reciprocal Rank)

### Output Files

```
output/
└── cosine_similarity/
    └── results.json  # Evaluation results in JSON format
```

**Example results.json:**

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
  }
]
```

---

## 3. cross_encoder.sh

### Overview

Fine-tunes and evaluates a cross-encoder model. By default, trains and evaluates on the Science subject.

### Prerequisites

- `preprocess.sh` execution completed
- Training and validation datasets prepared

### Configuration

```bash
TRAIN_CSV="${PROJECT_ROOT}/dataset/train_80/과학.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/valid_80/과학.csv"
```

### Execution

```bash
cd scripts
bash cross_encoder.sh
```

### Processing Pipeline

1. **Step 1: Cross-Encoder Fine-tuning**

   - Executes `finetune_cross_encoder.py`
   - Trains the model on training data
   - Output: `model/cross_finetuned/`

2. **Step 2: Cross-Encoder Evaluation**
   - Executes `eval_cross_encoder.py`
   - Retrieves candidates using a bi-encoder, then re-ranks with the cross-encoder
   - Evaluates on validation data
   - Output: `output/cross_encoder/results_rerank.json`

### Output Files

```
model/
└── cross_finetuned/           # Fine-tuned cross-encoder model
    ├── config.json
    ├── pytorch_model.bin
    └── ...

output/
└── cross_encoder/
    ├── results_rerank.json    # Evaluation results
    └── logs/
        └── 과학_wrongs.txt     # Misclassified sample logs (max 100 samples)
```

**Example results_rerank.json:**

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
  }
]
```

---

## 4. train_classifier.sh

### Overview

Trains a multi-class classifier using `klue/roberta-large` for achievement standard classification.

### Prerequisites

- `preprocess.sh` execution completed
- `dataset/train_80/` directory exists

### Execution

```bash
cd scripts
bash train_classifier.sh
```

### Default Settings

```
base_model: klue/roberta-large

epochs: 10
batch_size: 32

lr: 2e-5
weight_decay: 0.01
pooling: cls
dropout: 0.1

gradient_accumulation_steps: 1
warmup_ratio: 0.1
early_stopping_patience: 3
mixed_precision: True

loss_type: ce
label_smoothing: 0.1
focal_alpha: 1.0
focal_gamma: 2.0
```

### Output

- Trained classification model
- Training logs and evaluation metrics

---

## 5. train_advanced.sh

### Overview

Training script for an advanced multi-class classifier with enhanced training techniques.

### Prerequisites

- `preprocess.sh` execution completed

### Execution

```bash
cd scripts
bash train_advanced.sh
```

### Implementation Details

The script implements advanced training configurations for improved model performance. Refer to the source code for specific implementation details.

---

## 6. eval_classifier.sh

### Overview

Evaluation script for multi-class classifiers.

### Prerequisites

- `preprocess.sh` execution completed
- `train_classifier.sh` execution completed

### Execution

```bash
cd scripts
bash eval_classifier.sh
```

---

## 7. llm_text_classification.sh

### Overview

Text classification evaluation script using Large Language Models (LLMs). Sequentially processes all subjects in the validation dataset.

### Prerequisites

- `preprocess.sh` execution completed
- CUDA-enabled GPU (recommended)
- Sufficient VRAM (varies by model size)

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # LLM model to use
MAX_NEW_TOKENS=50                       # Maximum tokens to generate
TEMPERATURE=0.1                         # Sampling temperature (lower values are more deterministic)
DEVICE="cuda"                           # Device (cuda or cpu)
MAX_INPUT_LENGTH=6144                   # Maximum input length
NUM_SAMPLES=100                         # Target number of samples to evaluate (None for all)
```

### Execution

```bash
cd scripts
bash llm_text_classification.sh
```

```

### Methodology

1. Discovers all CSV files in the `valid_80` folder (9 subjects)
2. For each CSV file (subject), sequentially:
   - Loads few-shot examples for the subject (`few_shot_examples/{subject}.json`)
   - Loads the LLM
   - Predicts achievement standards using few-shot prompting
   - Computes accuracy and MRR
   - Saves correct/incorrect samples
3. Continues processing subsequent files even if errors occur

### Output Files

```

output/
└── llm_text_classification/
├── results.json # Evaluation results for all subjects
└── logs/
├── 과학\_corrects.txt # Subject-specific correct sample logs
├── 과학\_wrongs.txt # Subject-specific incorrect sample logs
├── 국어\_corrects.txt
├── 국어\_wrongs.txt
└── ...

````

**Example results.json:**

```json
[
  {
    "folder": "valid_80",
    "model_path": "Qwen/Qwen2.5-3B-Instruct",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "subject": "과학",
    "num_standards": 190,
    "num_candidates": 120,
    "max_candidates": 120,
    "num_samples": 100,
    "total_samples": 100,
    "correct": 65,
    "accuracy": 0.65,
    "mrr": 0.72,
    "exact_match_count": 58,
    "exact_match_percentage": 0.58,
    "match_type_distribution": {
      "exact": 58.0,
      "partial": 35.0,
      "invalid": 7.0
    },
    "max_new_tokens": 50,
    "temperature": 0.1,
    "max_input_length": 6144,
    "truncated_count": 0,
    "truncated_percentage": 0.0
  }
]
````

### Customization

#### Using a Different LLM Model

```bash
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
```

#### Running on CPU

```bash
DEVICE="cpu"
```

#### Evaluating the Entire Dataset

```bash
NUM_SAMPLES=None  # Or a very large number
```

#### Evaluating on Training Dataset

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/train_80"
```

---

## 8. finetuning_llm.sh

### Overview

Fine-tunes an LLM for the achievement standard classification task.

### Prerequisites

- `preprocess.sh` execution completed
- `dataset/train_80/` directory exists
- Sufficient VRAM for LLM fine-tuning

### Execution

```bash
cd scripts
bash finetuning_llm.sh
```

### Configuration

```bash
MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # Base model for fine-tuning
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_llm"   # Path to save fine-tuned model
```

### Methodology

1. Loads the training dataset
2. Loads the base LLM
3. Fine-tunes the model for the achievement standard classification task
4. Saves the fine-tuned model

### Output Files

```
model/
└── finetuned_llm/           # Fine-tuned LLM
    ├── config.json
    ├── model weights
    ├── tokenizer files
    └── training_args.json
```

### Implementation Details

- Instruction tuning approach
- Efficient fine-tuning techniques (LoRA, QLoRA, etc.) available
- Training logs and checkpoint saving

---

## 9. finetune_llm_text_classification.sh

### Overview

Evaluation script for fine-tuned LLMs. Sequentially processes all subjects in the validation dataset.

### Prerequisites

- `preprocess.sh` execution completed
- `finetuning_llm.sh` execution completed (fine-tuned model required)
- `model/finetuned_llm/` directory exists

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_PATH="${PROJECT_ROOT}/model/finetuned_llm"  # Path to fine-tuned model
MAX_NEW_TOKENS=50
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=6144
NUM_SAMPLES=100
```

### Execution

```bash
cd scripts
bash finetune_llm_text_classification.sh
```

### Methodology

1. Loads the fine-tuned LLM
2. Processes all CSV files in the `valid_80` folder (9 subjects)
3. For each subject:
   - Evaluates with few-shot examples
   - Saves correct/incorrect samples
   - Computes performance metrics

### Output Files

```
output/
└── llm_text_classification/
    ├── finetuned_results.json    # Fine-tuned LLM evaluation results
    └── finetuned_logs/
        ├── finetuned_llm_과학_corrects.txt
        ├── finetuned_llm_과학_wrongs.txt
        ├── finetuned_llm_국어_corrects.txt
        ├── finetuned_llm_국어_wrongs.txt
        └── ...
```

**Example finetuned_results.json:**

```json
[
  {
    "folder": "valid_80",
    "model_path": "/path/to/model/finetuned_llm",
    "base_model": "N/A",
    "subject": "과학",
    "num_standards": 190,
    "num_candidates": 120,
    "max_candidates": 120,
    "num_samples": 100,
    "total_samples": 100,
    "correct": 75,
    "accuracy": 0.75,
    "mrr": 0.82,
    "exact_match_count": 70,
    "exact_match_percentage": 0.7,
    "match_type_distribution": {
      "exact": 70.0,
      "partial": 25.0,
      "invalid": 5.0
    },
    "max_new_tokens": 50,
    "temperature": 0.1,
    "max_input_length": 6144,
    "truncated_count": 0,
    "truncated_percentage": 0.0,
    "training_info": {}
  }
]
```

### Implementation Details

- Measures fine-tuning effectiveness
- Enables performance comparison with pre-trained models
- Provides detailed logs and analysis materials

---

## 10. rag_llm_text_classification.sh

### Overview

Text classification evaluation script using RAG (Retrieval-Augmented Generation) workflow with Large Language Models. Implements a two-stage approach: first retrieves top-k candidate achievement standards using a multi-class classifier, then uses an LLM to select the best match from the candidates.

### Prerequisites

- `preprocess.sh` execution completed
- Trained multi-class classifier model available (`model/achievement_classifier/best_model`)
- CUDA-enabled GPU (recommended)
- Sufficient VRAM (varies by model size)

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_NAME="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # LLM model to use
MAX_NEW_TOKENS=10                                   # Maximum tokens to generate
TEMPERATURE=0.1                                     # Sampling temperature
DEVICE="cuda"                                       # Device (cuda or cpu)
MAX_INPUT_LENGTH=4000                               # Maximum input length
TOP_K=20                                            # Number of candidates to retrieve
NUM_SAMPLES=200                                     # Target number of samples to evaluate
NUM_EXAMPLES=5                                      # Number of few-shot examples
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"  # Retrieval model
INFER_DEVICE="cuda"                                 # Device for retrieval model
```

### Execution

```bash
cd scripts
bash rag_llm_text_classification.sh
```

### Methodology

1. Discovers all CSV files in the `valid_80` folder (9 subjects)
2. For each CSV file (subject), sequentially:
   - **Step 1**: Retrieves top-k candidate achievement standards using `infer_top_k` from the trained multi-class classifier
   - **Step 2**: Loads few-shot examples for the subject
   - **Step 3**: Loads the LLM
   - **Step 4**: LLM selects the best matching achievement standard from the retrieved candidates
   - Computes accuracy and MRR
   - Saves correct/incorrect samples
3. Continues processing subsequent files even if errors occur

### Output Files

```
output/
└── rag_llm_text_classification/
    ├── results.json           # Evaluation results for all subjects
    └── logs/
        ├── {subject}_corrects.txt
        ├── {subject}_wrongs.txt
        └── ...
```

**Example results.json:**

```json
[
  {
    "folder": "valid_80",
    "model_path": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "subject": "과학",
    "num_standards": 190,
    "top_k": 20,
    "num_samples": 200,
    "total_samples": 200,
    "correct": 150,
    "accuracy": 0.75,
    "exact_match_count": 145,
    "exact_match_percentage": 0.725,
    "match_type_distribution": {
      "exact": 72.5,
      "partial": 25.0,
      "invalid": 2.5
    },
    "max_new_tokens": 10,
    "temperature": 0.1,
    "max_input_length": 4000,
    "truncated_count": 0,
    "truncated_percentage": 0.0
  }
]
```

### Customization

#### Using a Different LLM Model

```bash
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
```

#### Adjusting Retrieval Parameters

```bash
TOP_K=30  # Retrieve more candidates
```

#### Running on CPU

```bash
DEVICE="cpu"
```

---

## 11. api_rag_llm_text_classification.sh

### Overview

API-based RAG LLM text classification evaluation script. Uses external API providers (OpenRouter, OpenAI, Anthropic, Google) instead of local models for the LLM component, while still using a local multi-class classifier for candidate retrieval.

### Prerequisites

- `preprocess.sh` execution completed
- Trained multi-class classifier model available (`model/achievement_classifier/best_model`)
- API key configured in `.env` file or provided via environment variables

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
API_PROVIDER="openrouter"                    # API provider (openrouter, openai, anthropic, google)
API_MODEL="qwen/qwen-2.5-7b-instruct:free"  # API model name
API_DELAY=1.0                                # Delay between API calls (seconds)
TOP_K=20                                     # Number of candidates to retrieve
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"
INFER_DEVICE="cuda"
MAX_NEW_TOKENS=20
TEMPERATURE=0.1
NUM_SAMPLES=200
NUM_EXAMPLES=5
```

### Execution

```bash
cd scripts
bash api_rag_llm_text_classification.sh
```

### Methodology

Same two-stage RAG workflow as `rag_llm_text_classification.sh`, but uses API-based LLM instead of local model:

1. Retrieves top-k candidates using local multi-class classifier
2. Sends candidates to API-based LLM for final selection
3. Processes all subjects sequentially

### Output Files

Same structure as `rag_llm_text_classification.sh`:

```
output/
└── rag_llm_text_classification/
    ├── results.json
    └── logs/
```

### Customization

#### Using Different API Providers

```bash
API_PROVIDER="openai"
API_MODEL="gpt-4"
```

```bash
API_PROVIDER="anthropic"
API_MODEL="claude-3-5-sonnet-20241022"
```

---

## 12. rag_finetuning_llm.sh

### Overview

Fine-tunes an LLM for RAG-based achievement standard classification. The model is trained on the RAG workflow where it receives retrieved candidates and learns to select the best match.

### Prerequisites

- `preprocess.sh` execution completed
- `dataset/train_80/` directory exists
- Trained multi-class classifier model available (`model/achievement_classifier/best_model`)
- Sufficient VRAM for LLM fine-tuning

### Configuration

```bash
TRAIN_DIR="${PROJECT_ROOT}/dataset/train_80"
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"
MODEL_NAME="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_rag_llm"
MAX_SEQ_LENGTH=2600
NUM_SAMPLES=2500                    # Target number of samples per CSV file
TOP_K=20                            # Number of candidates to retrieve
INFER_DEVICE="cuda"
NUM_EXAMPLES_FEW_SHOT=5             # Number of few-shot examples
```

### Execution

```bash
cd scripts
bash rag_finetuning_llm.sh
```

### Methodology

1. Loads training dataset from `train_80/` directory
2. For each training CSV file:
   - Retrieves top-k candidates using multi-class classifier
   - Generates training prompts with RAG workflow
3. Fine-tunes the LLM on RAG-based classification task
4. Saves the fine-tuned model

### Output Files

```
model/
└── finetuned_rag_llm/           # Fine-tuned RAG LLM
    ├── config.json
    ├── model weights
    ├── tokenizer files
    └── training_args.json
```

### Implementation Details

- Instruction tuning approach with RAG workflow
- Efficient fine-tuning techniques (LoRA, QLoRA, etc.)
- Training logs and checkpoint saving

---

## 13. rag_finetune_llm_text_classification.sh

### Overview

Evaluation script for fine-tuned RAG LLMs. Sequentially processes all subjects in the validation dataset using the RAG workflow.

### Prerequisites

- `preprocess.sh` execution completed
- `rag_finetuning_llm.sh` execution completed (fine-tuned RAG model required)
- `model/finetuned_rag_llm/` directory exists
- Trained multi-class classifier model available

### Configuration

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_PATH="${PROJECT_ROOT}/model/finetuned_rag_llm/{checkpoint}"  # Path to fine-tuned RAG model
MODEL_DIR="${PROJECT_ROOT}/model/achievement_classifier/best_model"
MAX_NEW_TOKENS=20
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=2600
TOP_K=20
NUM_SAMPLES=200
NUM_EXAMPLES=0                      # Few-shot examples (0 to disable)
INFER_DEVICE="cuda"
```

### Execution

```bash
cd scripts
bash rag_finetune_llm_text_classification.sh
```

### Methodology

1. Loads the fine-tuned RAG LLM
2. Processes all CSV files in the `valid_80` folder (9 subjects)
3. For each subject:
   - Retrieves top-k candidates using multi-class classifier
   - Evaluates with fine-tuned RAG LLM
   - Saves correct/incorrect samples
   - Computes performance metrics

### Output Files

```
output/
└── rag_llm_text_classification/
    ├── finetuned_results.json    # Fine-tuned RAG LLM evaluation results
    └── finetuned_logs/
        ├── {subject}_corrects.txt
        ├── {subject}_wrongs.txt
        └── ...
```

**Example finetuned_results.json:**

Similar format to `rag_llm_text_classification.sh` results, with additional training information.

### Implementation Details

- Measures fine-tuning effectiveness for RAG workflow
- Enables performance comparison with pre-trained RAG LLMs
- Provides detailed logs and analysis materials

---

## Recommended Execution Order

For executing the complete pipeline from scratch:

### Basic Pipeline

```bash
cd scripts

# Step 1: Data preprocessing
bash preprocess.sh

# Step 2: Cosine similarity baseline evaluation
bash cosine_similarity.sh

# Step 3: Cross-encoder training and evaluation
bash cross_encoder.sh

# Step 4: LLM-based classification evaluation
bash llm_text_classification.sh
```

### Multi-class Classifier Training Pipeline

```bash
cd scripts

# Data preprocessing (skip if already completed)
bash preprocess.sh

# Train various classifiers
bash train_classifier.sh              # Basic multi-class classifier
bash eval_classifier.sh               # Evaluate multi-class classifier
bash train_advanced.sh                 # Advanced multi-class classifier
```

### LLM Fine-tuning Pipeline

```bash
cd scripts

# Data preprocessing (skip if already completed)
bash preprocess.sh

# LLM fine-tuning and evaluation
bash finetuning_llm.sh                        # LLM fine-tuning
bash finetune_llm_text_classification.sh      # Fine-tuned LLM evaluation

# Pre-trained LLM evaluation for comparison
bash llm_text_classification.sh               # Pre-trained LLM evaluation
```

### RAG LLM Pipeline

```bash
cd scripts

# Data preprocessing (skip if already completed)
bash preprocess.sh

# Train multi-class classifier (required for RAG retrieval)
bash train_classifier.sh

# RAG LLM evaluation
bash rag_llm_text_classification.sh           # Pre-trained RAG LLM evaluation
bash api_rag_llm_text_classification.sh        # API-based RAG LLM evaluation

# RAG LLM fine-tuning and evaluation
bash rag_finetuning_llm.sh                   # RAG LLM fine-tuning
bash rag_finetune_llm_text_classification.sh # Fine-tuned RAG LLM evaluation
```

## Common Considerations

### Path Configuration

- All scripts automatically detect the project root
- Can be executed from any directory

### Output Format

- All scripts support colored terminal output
- Displays progress and results in real-time

### Error Handling

- **Immediate termination**: `preprocess.sh`, `cosine_similarity.sh`, `cross_encoder.sh`, classifier training scripts
  - Terminates immediately on error (`set -e`)
- **Continue processing**: `llm_text_classification.sh`, `finetune_llm_text_classification.sh`, `rag_llm_text_classification.sh`, `api_rag_llm_text_classification.sh`, `rag_finetune_llm_text_classification.sh`
  - Continues processing subsequent files even on error (batch processing)

## Troubleshooting

### Dataset Not Found

```
Error: Dataset folder not found
```

→ Execute `preprocess.sh` first to generate the dataset
→ Or use `dataset/dataset_bundle.tar.gz` to create a sample dataset

---
