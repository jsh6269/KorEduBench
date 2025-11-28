# KorEduBench

Korean Education Benchmark - Educational achievement standard classification system.

## Description

KorEduBench is a benchmark project for 2022 Korean educational achievement standard classification. It provides various methods for automatically mapping textbook texts to achievement standards:

- **Embedding-based**: Cosine similarity, Cross-encoder
- **Classification-based**: Multi-class classifier
- **LLM-based**: Zero-shot, Few-shot, Fine-tuned LLM (with Unsloth support), RAG-based approach

## Requirements

- Python >= 3.11
- See `pyproject.toml` for full dependencies

## Installation

### Install uv (Recommended Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

For more installation options, see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Local Environment

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### API Key Configuration (Optional)

If you plan to use API-based models (OpenAI, Anthropic, Google AI Studio, OpenRouter, etc.):

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-your-actual-key-here
# GOOGLE_API_KEY=your-google-api-key-here
# OPENROUTER_API_KEY=your-openrouter-key-here
```

**Usage in code:**

```python
from src.utils.env_loader import get_api_key

# Get API key (returns None if not found)
openai_key = get_api_key("OPENAI_API_KEY")

# Or require the key (raises error if not found)
openai_key = get_api_key("OPENAI_API_KEY", required=True)
```

The `.env` file is automatically ignored by git to keep your keys secure.

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

Since processing the dataset may take a long time, we've provided a well‑organized, pre‑processed dataset at `dataset/dataset_bundle.tar.gz`. Once you extract this archive in the `dataset/` directory, you may skip `#0` and `#1`.

### 0. Download Dataset

Download [Curriculum-Level Subject Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71855)  
Note that we only use texts (not the images) which means **label directory** of dataset above is used in our project.

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

# Multi-class classifier
bash train_classifier.sh
bash eval_classifier.sh

# LLM evaluation
bash llm_text_classification.sh

# LLM fine-tuning
bash finetuning_llm.sh
bash finetune_llm_text_classification.sh
```

### Detailed Usage

For detailed script usage, see [`docs/SCRIPTS.md`](docs/SCRIPTS.md).

## Project Structure

```
KorEduBench/
├── src/                             # Source code
│   ├── preprocessing/               # Data preprocessing
│   ├── cosine_similarity/           # Embedding-based evaluation
│   ├── cross_encoder/               # Cross-encoder
│   ├── classification/              # Classifier training
│   └── llm_text_classification/     # LLM-based classification
│   └── rag_llm_text_classification/ # RAG based LLM classification
├── dataset/                         # Datasets
│   ├── train_80/                    # Train data by subject (80 texts)
│   ├── valid_80/                    # Validation data by subject
│   └── few_shot_examples/           # Few-shot examples
├── model/                           # Trained models
├── output/                          # Evaluation results
├── scripts/                         # Execution scripts (11 scripts)
└── docs/                            # Documentation
    ├── PROJECT_STRUCTURE.md         # Detailed project structure
    └── SCRIPTS.md                   # Script usage guide
```

## Dataset

- **Source**: [AI Hub - Curriculum-Level Subject Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71855)

- **Overview**
  This dataset is designed to support research in curriculum-aligned natural language understanding and multimodal learning. It was constructed through the systematic collection of textual and visual data from official educational materials, such as textbooks and reference guides, across multiple educational stages. These resources were then rigorously annotated and aligned with the achievement standards defined in the 2022 Revised National Curriculum of Korea, across nine core subject domains. The resulting dataset facilitates a range of educational AI tasks, including curriculum-based content inference, standard-level classification, and subject-specific knowledge modeling.

- **Subjects**:
  Science, Korean, Mathematics, English, Social Studies, Sociology, Ethics, Technology–Home Economics, Information (9 subjects in total)

- **Split**:
  The dataset is partitioned into training and validation sets, each containing 80 textual samples per achievement standard to ensure balanced representation across labels.

- **Contributors**:
  Media Group Sarangwasup Co., Ltd.

## Documentation

- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**: Project structure and data flow
- **[SCRIPTS.md](docs/SCRIPTS.md)**: Detailed usage guide for 11 scripts
- **[CODE_ANALYSIS.md](docs/CODE_ANALYSIS.md)**: Code analysis and technical documentation
