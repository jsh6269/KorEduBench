# KorEduBench

Korean Education Benchmark - Educational achievement standard classification system.

## Description

KorEduBench is a benchmark project for Korean educational achievement standard classification. It provides various methods for automatically mapping textbook texts to achievement standards:

- **Embedding-based**: Cosine similarity, Cross-encoder
- **Classification-based**: Multi-class classifiers (Focal loss, advanced techniques)
- **LLM-based**: Zero-shot, Few-shot, Fine-tuned LLM (with Unsloth support)

## Requirements

- Python >= 3.11
- See `pyproject.toml` for full dependencies

## Installation

### Local Environment

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Google Colab

```bash
# Install uv in Colab (first time only)
!pip install uv

# Install project dependencies with Colab extras
!uv sync --extra colab
```

**Alternative: Using pip**

```python
# Step 1: Install base dependencies (excluding unsloth)
!pip install -r requirements_colab.txt

# Step 2: Install unsloth with proper dependencies
import os
import re

if "COLAB_" in "".join(os.environ.keys()):
    import torch
    v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
```

**⚠️ Important**: Use `requirements_colab.txt` (not `requirements.txt`) in Colab to avoid dependency conflicts with `unsloth`.

### Important: Package Versions

This project requires specific package versions for compatibility with Unsloth:
- `transformers==4.56.2` (newer versions may have issues)
- `trl==0.22.2` (required for SFTTrainer)
- `torch==2.8.0` with `xformers==0.0.32.post2`

All version conflicts are managed by `uv.lock` file.

## Quick Start

### 1. Data Preprocessing
```bash
cd scripts && bash preprocess.sh
```

### 2. Run Evaluation
```bash
# Cosine similarity baseline
bash cosine_similarity.sh

# Cross-encoder
bash cross_encoder.sh

# LLM evaluation
bash llm_text_classification.sh

# LLM fine-tuning
bash finetuning_llm.sh
bash finetune_llm_text_classification.sh
```

### Detailed Usage
For detailed script usage, see [`doc/SCRIPTS.md`](doc/SCRIPTS.md).

## Project Structure

```
KorEduBench/
├── src/                         # Source code
│   ├── preprocessing/           # Data preprocessing
│   ├── cosine_similarity/       # Embedding-based evaluation
│   ├── cross_encoder/           # Cross-encoder
│   ├── classification/          # Classifier training
│   └── llm_text_classification/ # LLM-based classification
├── dataset/                     # Datasets
│   ├── train_80/               # Train data by subject (80 texts)
│   ├── valid_80/               # Validation data by subject
│   └── few_shot_examples/      # Few-shot examples
├── model/                       # Trained models
├── output/                      # Evaluation results
├── scripts/                     # Execution scripts (11 scripts)
└── doc/                         # Documentation
    ├── PROJECT_STRUCTURE.md    # Detailed project structure
    └── SCRIPTS.md              # Script usage guide
```

## Dataset

- **Source**: [AI Hub - Curriculum-Level Subject Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71855)
- **Subjects**: Science, Korean, Math, English, Social Studies, Sociology, Ethics, Technology-Home Economics, Information (9 subjects)
- **Split**: Train 80% / Validation 20%
- **Samples**: Up to 80 texts per achievement standard

## Documentation

- **[PROJECT_STRUCTURE.md](doc/PROJECT_STRUCTURE.md)**: Project structure and data flow
- **[SCRIPTS.md](doc/SCRIPTS.md)**: Detailed usage guide for 11 scripts
- **[CODE_ANALYSIS.md](doc/CODE_ANALYSIS.md)**: Code analysis and technical documentation

## License

See LICENSE file for details.

