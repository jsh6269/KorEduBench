# KorEduBench

Korean Education Benchmark - Educational achievement standard classification system.

## Description

This project provides tools for classifying Korean educational textbook excerpts into achievement standards using various methods including:
- Cosine similarity with embeddings
- Cross-encoder models
- LLM-based classification (with Unsloth fine-tuning support)

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

## Usage

### LLM Fine-tuning

```bash
python src/llm_text_classification/finetune_llm.py \
    --train_csv dataset/train.csv \
    --model_name unsloth/Qwen2.5-1.5B-Instruct \
    --num-train-epochs 3
```

### LLM Evaluation

```bash
python src/llm_text_classification/eval_llm.py \
    --input_csv dataset/valid.csv \
    --model_name Qwen/Qwen2.5-1.5B-Instruct
```

## Project Structure

```
KorEduBench/
├── src/
│   ├── llm_text_classification/
│   │   ├── finetune_llm.py
│   │   └── eval_llm.py
│   └── utils/
│       ├── data_loader.py
│       ├── prompt.py
│       └── random_seed.py
├── dataset/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── output/
    └── llm_text_classification/
```

## License

See LICENSE file for details.

