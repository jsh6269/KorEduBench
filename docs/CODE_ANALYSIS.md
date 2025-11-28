# KorEduBench Code Analysis

This document provides a comprehensive technical analysis of the KorEduBench project, detailing the implementation and functionality of each code module with step-by-step explanations.

---

## Table of Contents

1. [Data Preprocessing Pipeline](#1-data-preprocessing-pipeline)

   - 1.1 [extract_standards.py](#11-extract_standardspy)
   - 1.2 [add_text_to_standards.py](#12-add_text_to_standardspy)
   - 1.3 [verify_not_empty.py](#13-verify_not_emptypy)
   - 1.4 [check_insufficient_text.py](#14-check_insufficient_textpy)
   - 1.5 [add_additional_text_to_standards.py](#15-add_additional_text_to_standardspy)
   - 1.6 [filter_standards.py](#16-filter_standardspy)
   - 1.7 [split_subject.py](#17-split_subjectpy)

2. [Cosine Similarity Evaluation](#2-cosine-similarity-evaluation)

   - 2.1 [eval_cosine_similarity.py](#21-eval_cosine_similaritypy)
   - 2.2 [batch_cosine_similarity.py](#22-batch_cosine_similaritypy)

3. [Cross-Encoder Training and Evaluation](#3-cross-encoder-training-and-evaluation)

   - 3.1 [finetune_cross_encoder.py](#31-finetune_cross_encoderpy)
   - 3.2 [eval_cross_encoder.py](#32-eval_cross_encoderpy)

4. [Multi-Class Classifier Training and Evaluation](#4-multi-class-classifier-training-and-evaluation)

   - 4.1 [train_multiclass_classifier.py](#41-train_multiclass_classifierpy)
   - 4.2 [eval_multiclass_classifier.py](#42-eval_multiclass_classifierpy)
   - 4.3 [predict_multiclass.py](#43-predict_multiclasspy)
   - 4.4 [inference.py](#44-inferencepy)

5. [LLM-based Text Classification](#5-llm-based-text-classification)

   - 5.1 [eval_llm.py](#51-eval_llmpy)
   - 5.2 [finetune_llm.py](#52-finetune_llmpy)
   - 5.3 [eval_finetune_llm.py](#53-eval_finetune_llmpy)

6. [RAG-based LLM Text Classification](#6-rag-based-llm-text-classification)
   - 6.1 [rag_eval_llm.py](#61-rag_eval_llmpy)
   - 6.2 [rag_finetune_llm.py](#62-rag_finetune_llmpy)
   - 6.3 [rag_eval_ft_llm.py](#63-rag_eval_ft_llmpy)

---

## 1. Data Preprocessing Pipeline

### 1.1 extract_standards.py

**Location**: `src/preprocessing/extract_standards.py`

**Purpose**: Extracts 2022 achievement standards from AI Hub dataset ZIP files and stores them in CSV format.

#### Technical Analysis

**Function**: `extract_unique_standards(label_dir, output_csv)`

**Processing Pipeline**:

1. **ZIP File Collection**

   - Recursively traverses `label_dir` directory using `os.walk()`
   - Collects all `.zip` files from all subdirectories

2. **ZIP File Processing**
   - Iterates through JSON files within each ZIP archive
   - Extracts the following information from JSON:
     - `source_data_info.2022_achievement_standard`: List of achievement standards
     - `raw_data_info`: Subject, school level, and grade metadata
3. **Achievement Standard Parsing**

   - Format: `"[code] content"` (e.g., `"[6과01-01] 물체의 운동을 관찰하여..."`)
   - Extracts code from text between `[` and `]`
   - Extracts content from text after `]`
   - Deduplication: Uses `unique_standards` dictionary with code as key to ensure uniqueness

4. **CSV Export**
   - Columns: `subject`, `school`, `grade`, `code`, `content`
   - Sorting: By subject, then by code
   - Encoding: UTF-8-sig (Excel-compatible)

**Input**:

- `label_dir`: Directory path containing ZIP files

**Output**:

- `unique_achievement_standards.csv`: Unique achievement standards list

**Example Execution**:

```bash
python extract_standards.py ./Training/label
```

---

### 1.2 add_text_to_standards.py

**Location**: `src/preprocessing/add_text_to_standards.py`

**Purpose**: Matches and appends actual educational text samples to extracted achievement standards CSV.

#### Technical Analysis

**Function**: `append_texts_to_csv(label_dir, csv_path, output_csv, max_texts)`

**Processing Pipeline**:

1. **CSV Loading and Preparation**

   - Auto-detects CSV encoding using `chardet`
   - Loads CSV into pandas DataFrame
   - Creates `text_1`, `text_2`, ..., `text_N` columns up to `max_texts`
   - Appends to existing `text_` columns if present

2. **Code-to-Index Mapping**

   - Creates `code_to_idx`: Maps achievement standard codes to DataFrame row indices
   - Hashmap structure for O(1) lookup performance

3. **ZIP File Iteration and Text Extraction**

   - Extracts learning data from each JSON file:
     - `learning_data_info.text_description`: Text description
     - `learning_data_info.text_qa`: Question-answer text
     - `learning_data_info.text_an`: Additional text
   - Concatenates three text fields with whitespace to create `combined_text`

4. **Text-to-Standard Matching**

   - Extracts code from JSON's `2022_achievement_standard`
   - If code exists in CSV, appends text to first empty `text_` column
   - Skips if `max_texts` columns are already filled

5. **Result Persistence**
   - Saves updated DataFrame to CSV
   - Uses UTF-8-sig encoding

**Input**:

- `label_dir`: ZIP file directory (Training/label)
- `csv_path`: Input CSV (standards-only file)
- `max_texts`: Maximum number of text samples to add (default: 160)

**Output**:

- `text_achievement_standards.csv`: CSV with text samples appended

**Example Execution**:

```bash
python add_text_to_standards.py ./Training/label --csv_path unique_achievement_standards.csv --max_texts 160
```

---

### 1.3 verify_not_empty.py

**Location**: `src/preprocessing/verify_not_empty.py`

**Purpose**: Validates that `text_` columns are filled sequentially without gaps (ensures no empty columns between filled ones).

#### Technical Analysis

**Function**: `verify_text_order(input_csv)`

**Processing Pipeline**:

1. **CSV Loading**

   - Auto-detects encoding and loads DataFrame
   - Finds all `text_` columns and sorts numerically

2. **Order Validation**

   - For each row, finds first empty `text_` column
   - Verifies that subsequent columns contain no data
   - Records issues (e.g., `text_5` empty but `text_10` contains data)

3. **Result Output**
   - Prints validation pass/fail status
   - Subject-wise issue summary
   - Detailed output of first 10 issues

**Input**:

- `input_csv`: CSV file path to validate

**Output**:

- Console output of validation results
- Exit code (0: success, 1: failure)

**Example Execution**:

```bash
python verify_not_empty.py text_achievement_standards.csv
```

---

### 1.4 check_insufficient_text.py

**Location**: `src/preprocessing/check_insufficient_text.py`

**Purpose**: Identifies achievement standards with insufficient text samples and saves them to a separate CSV.

#### Technical Analysis

**Function**: `check_insufficient_text(input_csv, output_csv, min_texts)`

**Processing Pipeline**:

1. **CSV Loading and Analysis**

   - Auto-detects encoding and loads data
   - Identifies all `text_` columns

2. **Text Count Calculation**

   - Counts non-empty `text_` columns per row
   - Treats NaN, empty strings, and whitespace-only strings as empty

3. **Statistical Output**

   - Row distribution by text count (0, 1-20, 21-40, ...)
   - Subject/school level sufficient/insufficient statistics
   - Mean, median, min/max text counts

4. **Insufficient Row Filtering**

   - Selects rows with fewer than `min_texts` samples
   - Saves columns: `subject`, `school`, `grade`, `code`, `content`, `text_count`

5. **Result Persistence**
   - Saves to `insufficient_text.csv`
   - Prints subject and school level statistics

**Input**:

- `input_csv`: Input CSV file (default: `text_achievement_standards.csv`)
- `min_texts`: Minimum number of text samples (default: 160)

**Output**:

- `insufficient_text.csv`: List of achievement standards with insufficient text samples

**Example Execution**:

```bash
python check_insufficient_text.py --min_texts 160
```

---

### 1.5 add_additional_text_to_standards.py

**Location**: `src/preprocessing/add_additional_text_to_standards.py`

**Purpose**: Adds additional text samples from Validation dataset to achievement standards listed in `insufficient_text.csv`.

#### Technical Analysis

**Function**: `add_additional_texts_to_csv(...)`

**Processing Pipeline**:

1. **Insufficient Standards Loading**

   - Loads `insufficient_text.csv`
   - Creates set of (code, content) tuples for O(1) lookup

2. **Existing CSV Loading**

   - Loads `text_achievement_standards.csv`
   - Creates additional `text_` columns if needed

3. **Validation ZIP File Iteration**

   - Processes ZIP files in Validation/label directory
   - Extracts `text_description` from JSON
   - Checks if achievement standard code is in insufficient list

4. **Text Addition**

   - Adds text only to standards in insufficient list
   - Appends to first empty `text_` column
   - Removes standard from insufficient list when `max_texts` is reached

5. **Result Persistence**
   - Overwrites original CSV file
   - Prints number of texts added

**Input**:

- `label_dir`: Validation ZIP file directory
- `insufficient_csv`: Insufficient standards list
- `text_standards_csv`: CSV to update
- `max_texts`: Maximum number of text columns

**Output**:

- `text_achievement_standards.csv` (updated)

**Example Execution**:

```bash
python add_additional_text_to_standards.py ./Validation/label --max_texts 160
```

---

### 1.6 filter_standards.py

**Location**: `src/preprocessing/filter_standards.py`

**Purpose**: Filters achievement standards with sufficient text samples and performs train/validation split.

#### Technical Analysis

**Function**: `split_texts_to_train_valid(...)`

**Processing Pipeline**:

1. **CSV Loading**

   - Loads `text_achievement_standards.csv`
   - Verifies `text_1` ~ `text_{num_texts}` range

2. **Filtering**

   - Counts non-empty columns in `text_1` ~ `text_{num_texts}` range per row
   - Selects rows with at least `num_texts` samples

3. **Text Preprocessing**

   - HTML table processing: Extracts content from `<td>` tags only
   - Converts line breaks to spaces
   - Normalizes multiple spaces to single space

4. **Train/Validation Split**

   - Randomly samples `num_texts` texts from each row
   - First `num_texts/2` samples → train
   - Remaining `num_texts/2` samples → validation
   - **Critical**: Train and validation texts for the same standard are completely separated

5. **Result Persistence**
   - `train.csv`: `num_texts/2` texts per achievement standard
   - `valid.csv`: `num_texts/2` texts per achievement standard
   - Structure: metadata columns + `text_1` ~ `text_{num_texts/2}`

**Input**:

- `input_csv`: Input CSV (default: `text_achievement_standards.csv`)
- `num_texts`: Number of texts to sample (must be even, default: 160)
- `train_csv`: Train output path
- `valid_csv`: Validation output path
- `seed`: Random seed (default: 42)

**Output**:

- `train.csv`: Training data
- `valid.csv`: Validation data

**Example Execution**:

```bash
python filter_standards.py --num_texts 160 --input_csv text_achievement_standards.csv --train_csv train.csv --valid_csv valid.csv
```

---

### 1.7 split_subject.py

**Location**: `src/preprocessing/split_subject.py`

**Purpose**: Splits `train.csv` and `valid.csv` by subject and saves as individual files.

#### Technical Analysis

**Function**: `split_csv_by_subject(input_path, output_folder, max_texts, encoding)`

**Processing Pipeline**:

1. **Output Directory Creation**

   - Format: `{output_folder}` (e.g., `train_80`, `valid_80`)
   - Creates directory if it doesn't exist (`exist_ok=True`)

2. **CSV Loading and Column Selection**

   - Base columns: `subject`, `school`, `grade`, `code`, `content`
   - Text columns: `text_1` ~ `text_{max_texts}`
   - Selects only existing columns (handles partial presence)

3. **Subject-wise Grouping and Persistence**
   - Groups data by subject using `df.groupby("subject")`
   - Sanitizes filenames: alphanumeric and `_` only, spaces converted to `_`
   - Saves each subject as `{subject_name}.csv`
   - Examples: `과학.csv`, `수학.csv`, `영어.csv`

**Input**:

- `input_path`: Unified CSV file path (`train.csv` or `valid.csv`)
- `output_folder`: Output folder name (e.g., `train_80`)
- `max_texts`: Number of text columns to include
- `encoding`: CSV encoding (default: utf-8-sig)

**Output**:

- Subject-specific CSV files in `{output_folder}/` directory

**Example Execution**:

```bash
python split_subject.py --input train.csv --output train_80 --max-texts 80
python split_subject.py --input valid.csv --output valid_80 --max-texts 80
```

---

## 2. Cosine Similarity Evaluation

### 2.1 eval_cosine_similarity.py

**Location**: `src/cosine_similarity/eval_cosine_similarity.py`

**Purpose**: Evaluates retrieval performance using bi-encoder-based cosine similarity (baseline approach).

#### Technical Analysis

**Function**: `evaluate_cosine_similarity_baseline(...)`

**Processing Pipeline**:

1. **Data Preparation**

   - Auto-detects CSV encoding (`detect_encoding`)
   - Validates required columns: `code`, `content`
   - Identifies `text_` columns (evaluation samples)
   - Auto-calculates `max_samples_per_row` (maximum texts per row)

2. **Bi-Encoder Model Loading**

   - Loads `SentenceTransformer` model
   - Default model: `jhgan/ko-sroberta-multitask` (Korean sentence embedding)
   - Sets model to evaluation mode

3. **Achievement Standard Encoding**

   - Encodes all achievement standard `content` fields into vectors
   - Converts to GPU tensors (`convert_to_tensor=True`)
   - Displays progress bar (`show_progress_bar=True`)

4. **Sample Text Collection and Encoding**

   - Extracts texts from `text_1`, `text_2`, ... columns per row
   - Applies `max_samples_per_row` limit
   - Creates sample and ground truth code pairs
   - Encodes all sample texts into vectors

5. **Cosine Similarity Computation**

   - `util.cos_sim(emb_samples, emb_contents)`: Computes similarity matrix
   - Dimensions: `[num_samples, num_standards]`
   - Calculates similarity between each sample and all achievement standards

6. **Top-K Accuracy Computation**

   - Computes Top-1, 3, 10, 20, 40, 60 accuracy
   - For each sample, checks if ground truth is in top-k predictions
   - Accuracy = correct samples / total samples

7. **MRR (Mean Reciprocal Rank) Computation**

   - Identifies rank of ground truth for each sample
   - Reciprocal Rank = 1 / rank
   - MRR = average of all RR values
   - Higher MRR indicates ground truth ranked higher

8. **Result Persistence**
   - Saves results to JSON file (`results.json`)
   - Updates existing results or appends new entries
   - Stores: model name, subject, accuracy metrics, MRR

**Evaluation Metrics**:

- **Top-K Accuracy**: Proportion of samples where ground truth is in top-k predictions
- **MRR**: Mean reciprocal rank of ground truth (1.0 for rank 1, 0.5 for rank 2, 0.33 for rank 3...)

**Input**:

- `input_csv`: CSV file to evaluate
- `model_name`: Bi-encoder model name
- `max_samples_per_row`: Maximum samples per row

**Output**:

- `results.json`: Evaluation results

**Example Execution**:

```bash
python eval_cosine_similarity.py --input_csv ../dataset/valid_80/과학.csv
```

---

### 2.2 batch_cosine_similarity.py

**Location**: `src/cosine_similarity/batch_cosine_similarity.py`

**Purpose**: Executes cosine similarity evaluation in batch for all CSV files in a directory.

#### Technical Analysis

**Function**: `evaluate_folder(...)`

**Processing Pipeline**:

1. **CSV File Collection**

   - Finds all `.csv` files in `folder_path`
   - Prints warning if no files found

2. **Per-File Evaluation**
   - Displays progress with `tqdm`
   - Calls `evaluate_cosine_similarity_baseline` for each file
   - Continues processing on error (skips failed files)
   - Accumulates all results in the same JSON file

**Input**:

- `folder_path`: Folder containing CSV files
- `model_name`: Bi-encoder model to use
- `json_path`: JSON file path for results

**Output**:

- `results.json`: Evaluation results for all subjects

**Example Execution**:

```bash
python batch_cosine_similarity.py --folder_path ../dataset/valid_80/
```

---

## 3. Cross-Encoder Training and Evaluation

### 3.1 finetune_cross_encoder.py

**Location**: `src/cross_encoder/finetune_cross_encoder.py`

**Purpose**: Fine-tunes a cross-encoder model on Korean educational data.

#### Technical Analysis

**Function**: `fine_tune_cross_encoder(...)`

#### Cross-Encoder Architecture

- **Bi-Encoder**: Encodes texts separately then computes similarity (fast but lower accuracy)
- **Cross-Encoder**: Encodes two texts together to directly compute relevance score (slower but more accurate)

**Processing Pipeline**:

1. **Data Loading and Splitting**

   - Loads achievement standard data from CSV
   - Train/Test split (default 8:2)
   - Splits at row level to prevent data leakage

2. **Training Pair Generation** (`build_pairs_from_df`)

   **Positive Pair Generation**:

   - Matches each achievement standard (`content`) with its corresponding text samples
   - Format: `(text_sample, achievement_standard_content)` → label=1.0
   - Uses `pos_keys` set for deduplication

   **Negative Pair Generation**:

   - Randomly matches texts with achievement standards (unrelated pairs)
   - Format: `(text_sample, random_content)` → label=0.0
   - Validates no overlap with positive pairs
   - Negative pair count = positive pair count × `neg_ratio`
   - Maximum attempts: `num_neg * 20`

3. **Model Initialization**

   - Base model: `bongsoo/albert-small-kor-cross-encoder-v1`
   - Korean-optimized lightweight cross-encoder
   - `num_labels=1`: Outputs continuous relevance score (0~1)

4. **Training Configuration**

   - Creates DataLoader (batch processing)
   - Warmup steps: 10% of total steps
   - Learning rate schedule: gradual increase then decrease (training stabilization)

5. **Fine-tuning Execution**

   - Calls `model.fit()`
   - Configures epochs, learning rate, warmup
   - Displays progress
   - Saves trained model to `output_dir`

6. **Evaluation**
   - Performs predictions on test set
   - Binary classification with 0.5 threshold
   - Computes metrics:
     - **Accuracy**: Proportion of correctly classified pairs
     - **F1 Score**: Harmonic mean of precision and recall
     - **ROC-AUC**: Comprehensive classification performance metric (0.5=random, 1.0=perfect)

**Input**:

- `input_csv`: Training CSV file
- `base_model`: Pre-trained cross-encoder model
- `epochs`: Number of training epochs (default: 2)
- `batch_size`: Batch size (default: 8)

**Output**:

- `cross_finetuned/`: Fine-tuned model directory
- Console output: Accuracy, F1, ROC-AUC

**Example Execution**:

```bash
python finetune_cross_encoder.py --input_csv ../dataset/train_80/과학.csv --epochs 3
```

---

### 3.2 eval_cross_encoder.py

**Location**: `src/cross_encoder/eval_cross_encoder.py`

**Purpose**: Improves accuracy by extracting candidates with bi-encoder then re-ranking with cross-encoder.

#### Technical Analysis

**Function**: `evaluate_bi_cross_pipeline(...)`

#### Two-Stage Retrieval Pipeline

1. **Bi-Encoder**: Rapidly extracts top-k candidates
2. **Cross-Encoder**: Precisely re-ranks candidates

**Processing Pipeline**:

1. **Data and Model Preparation**

   - Loads and validates CSV
   - Loads bi-encoder (`jhgan/ko-sroberta-multitask`)
   - Loads cross-encoder (fine-tuned model)
   - Vectorizes achievement standards (bi-encoder)

2. **Sample Collection**

   - Extracts evaluation samples from `text_` columns
   - Stores with ground truth codes
   - Applies `max_samples_per_row` limit

3. **Stage 1: Bi-Encoder Retrieval**

   - Encodes all samples into vectors
   - Computes cosine similarity
   - Selects top `top_k` candidates per sample (default: 20)
   - Rapidly reduces candidate pool

4. **Stage 2: Cross-Encoder Re-ranking**

   For each sample:

   - Creates pairs of `top_k` candidates with query
   - Computes precise relevance scores with cross-encoder
   - Re-sorts by score
   - Records incorrect samples if top-1 prediction is wrong

5. **Evaluation Metric Computation**

   - **Top-1/3/10/20 Accuracy**: Proportion of ground truth in top-k predictions
   - **MRR**: Mean reciprocal rank
   - Generally outperforms bi-encoder alone

6. **Result Persistence**

   **JSON Logging**:

   - Model information, subject, accuracy, MRR
   - Updates existing results or appends new entries
   - Saves to `results_rerank.json`

   **Error Analysis**:

   - Saves random 100 incorrect samples
   - Stores:
     - Input text
     - Ground truth code and content
     - Predicted code and content
   - Saves to `logs/{subject}_wrongs.txt`

**Input**:

- `input_csv`: CSV file to evaluate
- `bi_model`: Bi-encoder model
- `cross_model`: Cross-encoder model (fine-tuned)
- `top_k`: Number of candidates to re-rank

**Output**:

- `results_rerank.json`: Evaluation results
- `logs/{subject}_wrongs.txt`: Incorrect sample analysis

**Example Execution**:

```bash
python eval_cross_encoder.py --input_csv ../dataset/valid_80/과학.csv --cross_model ./cross_finetuned
```

---

## 4. Multi-Class Classifier Training and Evaluation

### 4.1 train_multiclass_classifier.py

**Location**: `src/classification/train_multiclass_classifier.py`

**Purpose**: Trains a multi-class classifier that directly maps educational texts to achievement standard codes, treating each achievement standard as a distinct class.

#### Technical Analysis

**Function**: `train_classifier(...)`

#### Multi-Class Classification Architecture

This approach formulates the task as a multi-class classification problem where:

- Each achievement standard code represents a distinct class
- The model performs direct classification from text to achievement standard code
- Advantages over bi-encoder approaches:
  - Direct classification yields higher accuracy
  - Single forward pass enables fast inference
  - Enhanced semantic understanding through end-to-end training
  - Supports advanced techniques (label smoothing, focal loss, etc.)

**Processing Pipeline**:

1. **Data Preparation**

   - Loads CSV file containing achievement standards and text samples
   - Extracts all text samples from `text_` columns
   - Builds label mappings: `code_to_idx`, `idx_to_code`, `code_to_content`
   - Applies `max_samples_per_class` limit if specified
   - Performs train/validation split at row level to prevent data leakage

2. **Model Architecture Initialization**

   - Base model: `klue/roberta-large` (Korean language model)
   - Architecture components:
     - Pre-trained transformer encoder
     - Pooling layer (CLS token or mean pooling)
     - Dropout layer for regularization
     - Classification head with `num_classes` output dimensions
   - Model configuration:
     - `max_length`: Maximum sequence length (default: 256)
     - `dropout`: Dropout rate (default: 0.1)
     - `pooling`: Pooling strategy (`cls` or `mean`)

3. **Loss Function Selection**

   - **Cross-Entropy Loss** (`ce`): Standard classification loss
   - **Label Smoothing Cross-Entropy** (`label_smoothing`): Regularization technique to prevent overconfidence
   - **Focal Loss** (`focal`): Addresses class imbalance by down-weighting easy examples
     - Parameters: `focal_alpha` (class weighting), `focal_gamma` (focusing parameter)

4. **Training Configuration**

   - Optimizer: AdamW with weight decay
   - Learning rate: 2e-5 (default)
   - Learning rate schedule: Cosine annealing with warmup
   - Warmup ratio: 10% of total training steps
   - Gradient accumulation: Configurable for effective larger batch sizes
   - Mixed precision training: FP16/BF16 for memory efficiency

5. **Training Execution**

   - Iterates through training epochs
   - Computes loss and backpropagates gradients
   - Evaluates on validation set after each epoch
   - Implements early stopping based on validation performance
   - Saves best model checkpoint based on validation accuracy

6. **Model Evaluation**

   - Computes comprehensive metrics on test set:
     - **Top-K Accuracy**: Accuracy at different k values (1, 3, 5, 10, 20)
     - **Weighted F1 Score**: Class-weighted F1 score
     - **Macro F1 Score**: Unweighted mean F1 across classes
     - **Precision and Recall**: Per-class and weighted averages

7. **Model Persistence**
   - Saves trained model weights: `model.pt`
   - Saves configuration: `config.json`
   - Saves label mappings: `label_mappings.json`
   - Saves training history and metrics

**Input**:

- `input_csv`: Training CSV file
- `base_model`: Pre-trained transformer model (default: `klue/roberta-large`)
- `output_dir`: Directory to save trained model
- `max_samples_per_class`: Maximum samples per achievement standard
- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 2e-5)
- `loss_type`: Loss function type (`ce`, `focal`, `label_smoothing`)

**Output**:

- `model/achievement_classifier/`: Trained model directory
  - `model.pt`: Model weights
  - `config.json`: Model configuration
  - `label_mappings.json`: Code-to-index mappings

**Example Execution**:

```bash
python train_multiclass_classifier.py \
    --input_csv ../dataset/train_80/과학.csv \
    --base_model klue/roberta-large \
    --output_dir ../model/achievement_classifier \
    --epochs 10 \
    --loss_type ce
```

---

### 4.2 eval_multiclass_classifier.py

**Location**: `src/classification/eval_multiclass_classifier.py`

**Purpose**: Evaluates a trained multi-class classifier on test data and computes comprehensive performance metrics.

#### Technical Analysis

**Function**: `evaluate_model(...)`

**Processing Pipeline**:

1. **Model Loading**

   - Loads trained model from checkpoint directory
   - Restores model architecture from `config.json`
   - Loads label mappings from `label_mappings.json`
   - Initializes tokenizer

2. **Data Preparation**

   - Loads test CSV file
   - Prepares test dataset using same preprocessing as training
   - Creates DataLoader for batch processing

3. **Inference**

   - Processes test samples in batches
   - Computes logits for all classes
   - Applies softmax to obtain class probabilities
   - Extracts top-k predictions for each sample

4. **Metric Computation**

   - **Top-K Accuracy**: Computes accuracy at multiple k values
   - **Weighted F1**: Class-weighted F1 score accounting for class imbalance
   - **Macro F1**: Unweighted mean F1 across all classes
   - **Precision/Recall**: Per-class and weighted averages
   - **Classification Report**: Detailed per-class metrics

5. **Result Persistence**
   - Saves evaluation metrics to JSON
   - Generates classification report
   - Outputs confusion matrix (optional)

**Input**:

- `input_csv`: Test CSV file
- `model_dir`: Directory containing trained model
- `batch_size`: Batch size for evaluation

**Output**:

- Evaluation metrics (JSON format)
- Classification report
- Per-class performance statistics

**Example Execution**:

```bash
python eval_multiclass_classifier.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_dir ../model/achievement_classifier/best_model
```

---

### 4.3 predict_multiclass.py

**Location**: `src/classification/predict_multiclass.py`

**Purpose**: Provides inference functionality for trained multi-class classifiers, enabling prediction on new text samples.

#### Technical Analysis

**Function**: `predict_batch(...)`

**Processing Pipeline**:

1. **Model Loading**

   - Loads trained model and configuration
   - Initializes tokenizer
   - Restores label mappings

2. **Text Preprocessing**

   - Tokenizes input texts
   - Applies padding and truncation
   - Converts to model input format

3. **Batch Inference**

   - Processes texts in batches for efficiency
   - Computes logits for all classes
   - Applies softmax to obtain probability distribution

4. **Top-K Extraction**

   - Extracts top-k class predictions with probabilities
   - Maps class indices to achievement standard codes
   - Retrieves corresponding content descriptions

5. **Result Formatting**
   - Returns structured predictions with:
     - Predicted achievement standard codes
     - Corresponding probabilities
     - Content descriptions

**Input**:

- `model`: Trained AchievementClassifier model
- `tokenizer`: Tokenizer instance
- `texts`: List of input text strings
- `top_k`: Number of top predictions to return
- `batch_size`: Batch size for inference

**Output**:

- List of prediction dictionaries containing:
  - `top_k`: List of top-k predictions with codes, probabilities, and contents

**Example Usage**:

```python
from src.classification.predict_multiclass import load_model, predict_batch

model, tokenizer, config, mappings = load_model(model_dir, device)
results = predict_batch(model, tokenizer, texts, device, top_k=10)
```

---

### 4.4 inference.py

**Location**: `src/classification/inference.py`

**Purpose**: Provides a high-level inference interface for retrieving top-k achievement standards for a given text, with filtering capabilities based on training data.

#### Technical Analysis

**Function**: `infer_top_k(...)`

**Processing Pipeline**:

1. **Model and Data Loading**

   - Loads trained classifier model (or uses provided model instance)
   - Loads training CSV to obtain available achievement standards
   - Builds code-to-content mapping from training data

2. **Text Classification**

   - Performs inference on input text
   - Obtains probability distribution over all classes
   - Maps class indices to achievement standard codes

3. **Candidate Filtering**

   - Filters predictions to only include codes present in training CSV
   - This ensures retrieved candidates are from the training distribution
   - Retrieves content descriptions for filtered candidates

4. **Top-K Selection and Ordering**

   - Selects top-k candidates based on probability scores
   - If `random=True`: Shuffles results to prevent position bias
   - If `random=False`: Maintains probability-based ordering (highest first)

5. **Result Formatting**
   - Returns structured dictionary with:
     - Input text
     - Top-k achievement standards with codes, contents, and probabilities
     - Metadata (k value, total candidates considered)

**Key Features**:

- **Training Data Filtering**: Only returns candidates from training distribution
- **Flexible Ordering**: Supports both probability-ordered and shuffled outputs
- **Batch Processing Support**: Can be used with pre-loaded models for efficiency

**Input**:

- `text`: Input text string to classify
- `top_k`: Number of top predictions to return
- `train_csv`: Path to training CSV (for filtering and content mapping)
- `model_dir`: Path to trained model directory (or provide model instance)
- `random`: Whether to shuffle results (default: True)

**Output**:

- Dictionary containing:
  - `text`: Input text
  - `k`: Number of results returned
  - `top_k`: List of top-k predictions with codes, contents, and probabilities

**Example Usage**:

```python
from src.classification.inference import infer_top_k

result = infer_top_k(
    text="일차방정식의 풀이 방법을 이해하고 활용할 수 있다",
    top_k=20,
    train_csv="../dataset/train_80/수학.csv",
    model_dir="../model/achievement_classifier/best_model",
    random=False
)
```

**Integration with RAG Workflow**:
This function serves as the retrieval component in the RAG pipeline:

1. `infer_top_k` retrieves top-k candidate achievement standards
2. LLM selects the best match from the retrieved candidates
3. This two-stage approach combines efficient retrieval with precise selection

---

## 5. LLM-based Text Classification

### 5.1 eval_llm.py

**Location**: `src/llm_text_classification/eval_llm.py`

**Purpose**: Classifies educational texts into achievement standards using pre-trained generative LLMs.

#### Overview

Unlike bi-encoder or cross-encoder approaches, this method uses generative LLMs to directly output achievement standard codes.

**Approach**:

- Lists all candidate achievement standards in the prompt
- LLM directly outputs the most relevant achievement standard code
- Example: `10영03-04`

#### Technical Analysis

**Function**: `evaluate_llm_classification(...)`

**Processing Pipeline**:

1. **Data Loading**

   - Loads achievement standards and sample texts from CSV
   - `num_samples`: Target number of samples to evaluate
   - `max_total_samples`: Total sample limit (random sampling if exceeded)

2. **Candidate Preparation**

   - Prepares all achievement standards as candidate list
   - `max_candidates`: Candidate limit (default: 200)
   - Random sampling if number of standards exceeds limit
   - Format: `[(number, code, content), ...]`

3. **LLM Model Loading**

   - Default model: `Qwen/Qwen2.5-3B-Instruct`
   - Loads with float16 precision (GPU)
   - Sets to evaluation mode

4. **Prompt Generation**

   - Generates classification prompt for each sample
   - Prompt structure:

     ```
     Select the achievement standard code that best describes the following educational content.

     Educational Content:
     [Sample text]

     Candidate Achievement Standards:
     1. [6과01-01] 물체의 운동을 관찰하여...
     2. [6과01-02] 자석의 성질을 탐구하여...
     ...

     Answer Code:
     ```

5. **Prediction Generation**

   - LLM generates code for each sample
   - `max_new_tokens`: Maximum tokens to generate (default: 50)
   - `temperature`: Sampling temperature (default: 0.1, nearly greedy)
   - `max_input_length`: Maximum input length (auto-truncates if exceeded)

6. **Response Parsing**

   - Extracts achievement standard code from LLM output
   - Classifies match type:
     - **exact**: Exact code match (e.g., `10영03-04`)
     - **partial**: Partial match (e.g., `03-04` restored to `10영03-04`)
     - **invalid**: Invalid code

7. **Evaluation**

   - **Accuracy**: Correct prediction rate
   - **MRR**: Mean reciprocal rank (same as accuracy for single prediction)
   - **Exact Match %**: Proportion of exact matches
   - **Match Type Distribution**: exact/partial/invalid proportions

8. **Result Persistence**
   - JSON logging: `results.json`
   - Incorrect samples: `logs/{subject}_wrongs.txt` (max 100)
   - Correct samples: `logs/{subject}_corrects.txt` (max 100)

**Input**:

- `input_csv`: CSV file to evaluate
- `model_name`: LLM model name (default: `Qwen/Qwen2.5-3B-Instruct`)
- `num_samples`: Target number of samples
- `max_total_samples`: Maximum total samples
- `max_candidates`: Maximum candidate count (default: 200)
- `max_new_tokens`: Maximum tokens to generate (default: 50)
- `temperature`: Sampling temperature (default: 0.1)
- `max_input_length`: Maximum input length (default: 6144)

**Output**:

- `output/llm_text_classification/results.json`: Evaluation results
- `output/llm_text_classification/logs/{subject}_wrongs.txt`: Incorrect samples
- `output/llm_text_classification/logs/{subject}_corrects.txt`: Correct samples

**Example Execution**:

```bash
python eval_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --max-candidates 120 \
    --num-samples 100
```

---

### 5.2 finetune_llm.py

**Location**: `src/llm_text_classification/finetune_llm.py`

**Purpose**: Fine-tunes generative LLMs for educational achievement standard classification task.

#### Overview

Uses Unsloth library for efficient LLM fine-tuning. Employs LoRA (Low-Rank Adaptation) technique for memory-efficient training.

#### Technical Analysis

**Function**: `finetune_llm(...)`

**Processing Pipeline**:

1. **Training Data Preparation**

   - Loads all CSV files in `train_dir` (subject-specific files)
   - Extracts achievement standards and text samples from each file
   - `num_samples`: Number of samples per achievement standard (default: 1)
   - `max_total_samples`: Total training sample limit
   - `max_candidates`: Candidate limit per prompt (default: 120)

2. **Prompt Generation**

   - Generates prompts in same format as evaluation
   - Adds ground truth code as completion
   - Format:
     ```
     [Prompt]
     Answer Code:
     [Ground truth code]
     ```

3. **Model Loading (Unsloth)**

   - Base model: `unsloth/Qwen2.5-3B-Instruct`
   - Loads with 4-bit quantization (memory efficient)
   - `max_seq_length`: Maximum sequence length (default: 6144)

4. **LoRA Adapter Addition**

   - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
   - `lora_r`: LoRA rank (default: 16)
   - `lora_alpha`: LoRA alpha (default: 16)
   - `lora_dropout`: LoRA dropout (default: 0.0)

5. **Training Configuration**

   - Uses `SFTTrainer` (Supervised Fine-Tuning Trainer)
   - Hyperparameters:
     - `num_train_epochs`: Number of epochs (default: 1)
     - `per_device_train_batch_size`: Batch size (default: 4)
     - `gradient_accumulation_steps`: Gradient accumulation (default: 4)
     - `learning_rate`: Learning rate (default: 2e-4)
     - `warmup_steps`: Warmup steps (default: 5)
   - Optimizer: AdamW 8-bit
   - Mixed Precision: FP16 or BF16 (depending on GPU support)

6. **Training Execution**

   - Calls `trainer.train()`
   - Automatic progress output
   - Automatic checkpoint saving

7. **Model Persistence**
   - **LoRA Adapter**: `output_dir/` (lightweight, reusable)
   - **Merged 16-bit Model**: `output_dir/merged_16bit/` (inference, high quality)
   - **Merged 4-bit Model**: `output_dir/merged_4bit/` (inference, memory efficient)
   - **Training Information**: `output_dir/training_info.json`

**Input**:

- `train_dir`: Directory containing training CSV files
- `model_name`: Base LLM model (default: `unsloth/Qwen2.5-3B-Instruct`)
- `output_dir`: Output directory (default: `model/finetuned_llm`)
- `max_seq_length`: Maximum sequence length (default: 6144)
- `num_samples`: Target number of samples per achievement standard (default: 1)
- `max_total_samples`: Maximum total samples (default: None)
- `max_candidates`: Maximum candidate count (default: 120)
- Training hyperparameters

**Output**:

- `model/finetuned_llm/`: LoRA adapter
- `model/finetuned_llm/merged_16bit/`: Merged 16-bit model
- `model/finetuned_llm/merged_4bit/`: Merged 4-bit model
- `model/finetuned_llm/training_info.json`: Training information

**Example Execution**:

```bash
python finetune_llm.py \
    --train_dir ../dataset/train_80 \
    --model_name unsloth/Qwen2.5-3B-Instruct \
    --output_dir ../model/finetuned_llm \
    --num-train-epochs 1 \
    --num-samples 1 \
    --max-candidates 120
```

---

### 5.3 eval_finetune_llm.py

**Location**: `src/llm_text_classification/eval_finetune_llm.py`

**Purpose**: Evaluates fine-tuned LLMs for educational text classification.

#### Overview

Loads models trained with `finetune_llm.py` for evaluation. Uses nearly identical evaluation process as `eval_llm.py` but with different model loading mechanism.

#### Technical Analysis

**Function**: `evaluate_finetuned_llm(...)`

**Processing Pipeline**:

1. **Data Loading**

   - Same as `eval_llm.py`

2. **Fine-tuned Model Loading**

   - Uses Unsloth's `FastLanguageModel.from_pretrained()`
   - Loading options:
     - `use_merged=True` (default): Loads `merged_16bit/` model
     - `use_merged=False`: Loads LoRA adapter
   - `FastLanguageModel.for_inference()`: Inference optimization (2x faster)

3. **Training Information Loading**

   - Reads training configuration from `training_info.json`
   - Prints base model, training sample count, epochs, etc.

4. **Evaluation**

   - Same process as `eval_llm.py`
   - Prompt generation → LLM prediction → Response parsing → Evaluation

5. **Result Persistence**
   - JSON logging: `finetuned_results.json`
   - Incorrect samples: `finetuned_logs/{model_name}_{subject}_wrongs.txt`
   - Correct samples: `finetuned_logs/{model_name}_{subject}_corrects.txt`

**Input**:

- `input_csv`: CSV file to evaluate
- `model_path`: Fine-tuned model directory
- `use_merged`: Whether to use merged model (default: True)
- Remaining parameters same as `eval_llm.py`

**Output**:

- `output/llm_text_classification/finetuned_results.json`: Evaluation results
- `output/llm_text_classification/finetuned_logs/`: Incorrect/correct samples

**Example Execution**:

```bash
python eval_finetune_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_path ../model/finetuned_llm \
    --max-candidates 120 \
    --num-samples 100
```

---

## 6. RAG-based LLM Text Classification

### 6.1 rag_eval_llm.py

**Location**: `src/rag_llm_text_classification/rag_eval_llm.py`

**Purpose**: Evaluates LLM-based classification using Retrieval-Augmented Generation (RAG) workflow. Implements a two-stage approach: first retrieves top-k candidate achievement standards using a trained multi-class classifier, then uses an LLM to select the best match from the candidates.

#### Overview

The RAG workflow combines the efficiency of dense retrieval with the reasoning capability of generative LLMs. This approach addresses the limitation of including all candidates in the prompt by first filtering to a smaller, more relevant candidate set.

**Architecture**:

- **Stage 1 (Retrieval)**: Multi-class classifier retrieves top-k candidates using `infer_top_k`
- **Stage 2 (Generation)**: LLM selects the best matching achievement standard from retrieved candidates

#### Technical Analysis

**Function**: `evaluate_llm_classification(...)`

**Processing Pipeline**:

1. **Data Loading**

   - Loads achievement standards and sample texts from CSV
   - `num_samples`: Target number of samples to evaluate
   - Uses `load_evaluation_data` without `max_candidates` (RAG uses retrieval instead)

2. **Multi-class Classifier Loading**

   - Loads trained multi-class classifier from `model_dir`
   - Required for `infer_top_k` function
   - Loads model, tokenizer, config, and mappings

3. **Per-Sample Processing**

   For each sample:

   **Stage 1: Candidate Retrieval**

   - Calls `infer_top_k` with sample text
   - Retrieves top-k candidate achievement standards (default: 20)
   - Candidates are ranked by classifier confidence
   - Converts to `(rank, code, content)` tuple list

   **Stage 2: LLM Selection**

   - Generates RAG prompt with retrieved candidates
   - Includes few-shot examples if enabled
   - LLM selects best matching code from candidates
   - Parses response to extract predicted code

4. **Prompt Generation**

   - Uses `create_rag_chat_prompt` from `rag_prompt.py`
   - Prompt structure includes:
     - System prompt: Educational curriculum expert role
     - Few-shot examples (optional)
     - Textbook text
     - Candidate achievement standards (retrieved)
     - Output format instructions

5. **Response Parsing**

   - Uses `parse_llm_response` to extract achievement standard code
   - Match types: exact, partial, invalid
   - Confidence scoring for fuzzy matches

6. **Evaluation Metrics**

   - **Accuracy**: Correct prediction rate
   - **Exact Match Count**: Number of exact code matches
   - **Match Type Distribution**: exact/partial/invalid proportions
   - **Truncation Count**: Number of prompts truncated due to length

7. **Result Persistence**
   - JSON logging: `results.json` in model-specific subdirectory
   - Incorrect samples: `logs/{subject}_wrongs.txt` (random 100)
   - Correct samples: `logs/{subject}_corrects.txt` (random 100)

**Input**:

- `input_csv`: CSV file to evaluate
- `generate_fn`: Function to generate predictions
- `model_identifier`: Model name or API identifier
- `tokenizer`: Tokenizer for token length checking (optional for API mode)
- `num_samples`: Target number of samples
- `train_csv`: Path to train CSV for `infer_top_k`
- `model_dir`: Path to multi-class classifier model directory
- `top_k`: Number of candidates to retrieve (default: 20)
- `infer_device`: Device for retrieval model (default: "cuda")
- `num_examples`: Number of few-shot examples (default: 5)

**Output**:

- `output/rag_llm_text_classification/{model_name}_{date}/results.json`: Evaluation results
- `output/rag_llm_text_classification/{model_name}_{date}/logs/`: Sample logs

**Example Execution**:

```bash
python rag_eval_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_name unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --train-csv ../dataset/train_80/과학.csv \
    --model-dir ../model/achievement_classifier/best_model \
    --top-k 20 \
    --num-samples 200
```

---

### 6.2 rag_finetune_llm.py

**Location**: `src/rag_llm_text_classification/rag_finetune_llm.py`

**Purpose**: Fine-tunes LLMs for RAG-based achievement standard classification. The model is trained on the RAG workflow where it receives retrieved candidates and learns to select the best match.

#### Overview

Uses Unsloth library for efficient fine-tuning with LoRA. The training process incorporates the RAG retrieval step, ensuring the model learns to work with retrieved candidates rather than all possible standards.

#### Technical Analysis

**Function**: `finetune_llm(...)`

**Processing Pipeline**:

1. **Training Data Preparation**

   - Loads all CSV files from `train_dir` (subject-specific files)
   - For each CSV file:
     - Uses the file itself as `train_csv` for `infer_top_k` filtering
     - Loads samples using `load_evaluation_data`
     - Retrieves top-k candidates for each sample using `infer_top_k`
     - Generates RAG prompts with retrieved candidates

2. **RAG Prompt Generation**

   - Uses `prepare_rag_training_dataset` from `data_loader.py`
   - For each training sample:
     - Retrieves top-k candidates using multi-class classifier
     - Generates prompt with `create_rag_chat_prompt`
     - Includes few-shot examples if enabled
     - Adds ground truth code as completion

3. **Model Loading (Unsloth)**

   - Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (default)
   - Loads with 4-bit quantization
   - `max_seq_length`: Maximum sequence length (default: 2600)

4. **LoRA Adapter Configuration**

   - Target modules: attention and MLP layers
   - `lora_r`: LoRA rank (default: 16)
   - `lora_alpha`: LoRA alpha (default: 16)
   - `lora_dropout`: LoRA dropout (default: 0.0)

5. **Training Configuration**

   - Uses `SFTTrainer` for supervised fine-tuning
   - Hyperparameters:
     - `num_train_epochs`: Number of epochs (default: 1)
     - `per_device_train_batch_size`: Batch size (default: 4)
     - `gradient_accumulation_steps`: Gradient accumulation (default: 4)
     - `learning_rate`: Learning rate (default: 1e-4)
     - `warmup_steps`: Warmup steps (default: 5)
   - Optimizer: AdamW 8-bit
   - Mixed precision training

6. **Training Execution**

   - Processes training examples with RAG workflow
   - Each example includes retrieved candidates in the prompt
   - Model learns to select correct code from candidates
   - Saves checkpoints at specified intervals

7. **Model Persistence**
   - **LoRA Adapter**: `output_dir/` (lightweight)
   - **Merged Models**: 16-bit and 4-bit merged versions
   - **Training Information**: `training_info.json`

**Input**:

- `train_dir`: Directory containing training CSV files
- `model_dir`: Path to multi-class classifier for retrieval
- `model_name`: Base LLM model
- `output_dir`: Output directory for fine-tuned model
- `top_k`: Number of candidates to retrieve (default: 20)
- `infer_device`: Device for retrieval model
- `num_examples`: Number of few-shot examples (default: 5)
- Training hyperparameters

**Output**:

- `model/finetuned_rag_llm/`: Fine-tuned RAG LLM
- `model/finetuned_rag_llm/training_info.json`: Training information

**Example Execution**:

```bash
python rag_finetune_llm.py \
    --train_dir ../dataset/train_80 \
    --model_dir ../model/achievement_classifier/best_model \
    --model_name unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --output_dir ../model/finetuned_rag_llm \
    --top-k 20 \
    --num-train-epochs 1
```

---

### 6.3 rag_eval_ft_llm.py

**Location**: `src/rag_llm_text_classification/rag_eval_ft_llm.py`

**Purpose**: Evaluates fine-tuned RAG LLMs for educational text classification using the RAG workflow.

#### Overview

Loads models trained with `rag_finetune_llm.py` and evaluates them using the same two-stage RAG workflow. The fine-tuned model has learned to work effectively with retrieved candidates.

#### Technical Analysis

**Function**: `evaluate_finetuned_rag_llm(...)`

**Processing Pipeline**:

1. **Data Loading**

   - Same as `rag_eval_llm.py`

2. **Fine-tuned Model Loading**

   - Loads fine-tuned RAG LLM using Unsloth
   - Uses `load_finetuned_model` utility
   - Loads training information from `training_info.json`

3. **Multi-class Classifier Loading**

   - Loads trained classifier for candidate retrieval
   - Same as `rag_eval_llm.py`

4. **Evaluation Process**

   - For each sample:
     - Retrieves top-k candidates using `infer_top_k`
     - Generates RAG prompt with candidates
     - Fine-tuned LLM selects best match
     - Parses and evaluates response

5. **Result Persistence**
   - JSON logging: `finetuned_results.json`
   - Sample logs: `finetuned_logs/` directory
   - Includes training information in results

**Input**:

- `input_csv`: CSV file to evaluate
- `model_path`: Path to fine-tuned RAG model
- `train_csv`: Path to train CSV for retrieval
- `model_dir`: Path to multi-class classifier
- `top_k`: Number of candidates to retrieve
- Other parameters same as `rag_eval_llm.py`

**Output**:

- `output/rag_llm_text_classification/{model_name}_{date}/finetuned_results.json`: Evaluation results
- `output/rag_llm_text_classification/{model_name}_{date}/finetuned_logs/`: Sample logs

**Example Execution**:

```bash
python rag_eval_ft_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_path ../model/finetuned_rag_llm/251127 \
    --train-csv ../dataset/train_80/과학.csv \
    --model-dir ../model/achievement_classifier/best_model \
    --top-k 20 \
    --num-samples 200
```

---

## Complete Pipeline Summary

### Phase 1: Data Preparation

```
Raw ZIP files (Training/label)
    ↓ extract_standards.py
unique_achievement_standards.csv (standards only)
    ↓ add_text_to_standards.py (Training)
text_achievement_standards.csv (text samples added, some insufficient)
    ↓ verify_not_empty.py (optional)
Validation complete
    ↓ check_insufficient_text.py
insufficient_text.csv (insufficient standards list)
    ↓ add_additional_text_to_standards.py (Validation)
text_achievement_standards.csv (updated, additional texts added)
    ↓ check_insufficient_text.py (recheck)
Sufficiency confirmed
    ↓ filter_standards.py
train.csv + valid.csv (train/validation split)
    ↓ split_subject.py
train_80/과학.csv, 수학.csv, ... (subject-wise split)
valid_80/과학.csv, 수학.csv, ... (subject-wise split)
```

### Phase 2: Baseline Evaluation (Bi-Encoder)

```
valid_80/*.csv
    ↓ eval_cosine_similarity.py (or batch)
output/cosine_similarity/results.json
```

### Phase 3: Cross-Encoder Fine-tuning and Evaluation

```
train_80/*.csv
    ↓ finetune_cross_encoder.py
cross_finetuned/ (fine-tuned model)
    ↓ eval_cross_encoder.py
output/cross_encoder/results_rerank.json
output/cross_encoder/logs/*_wrongs.txt
```

### Phase 4: LLM Baseline Evaluation

```
valid_80/*.csv
    ↓ eval_llm.py
output/llm_text_classification/results.json
output/llm_text_classification/logs/*_wrongs.txt
output/llm_text_classification/logs/*_corrects.txt
```

### Phase 5: LLM Fine-tuning and Evaluation

```
train_80/*.csv
    ↓ finetune_llm.py
model/finetuned_llm/ (LoRA adapter)
model/finetuned_llm/merged_16bit/ (merged model)
model/finetuned_llm/merged_4bit/ (4-bit quantized model)
    ↓ eval_finetune_llm.py
output/llm_text_classification/finetuned_results.json
output/llm_text_classification/finetuned_logs/*_wrongs.txt
output/llm_text_classification/finetuned_logs/*_corrects.txt
```

### Phase 6: Multi-class Classifier Training (for RAG)

```
train_80/*.csv
    ↓ train_multiclass_classifier.py
model/achievement_classifier/best_model/ (trained classifier)
```

### Phase 7: RAG LLM Evaluation

```
valid_80/*.csv
    ↓ rag_eval_llm.py
output/rag_llm_text_classification/{model}_{date}/results.json
output/rag_llm_text_classification/{model}_{date}/logs/
```

### Phase 8: RAG LLM Fine-tuning and Evaluation

```
train_80/*.csv
    ↓ rag_finetune_llm.py
model/finetuned_rag_llm/ (fine-tuned RAG LLM)
    ↓ rag_eval_ft_llm.py
output/rag_llm_text_classification/{model}_{date}/finetuned_results.json
output/rag_llm_text_classification/{model}_{date}/finetuned_logs/
```

---

## Core Concepts

### 1. Achievement Standard

- Learning objectives defined in educational curriculum
- Format: `[code] content` (e.g., `[6과01-01] 물체의 운동 관찰...`)
- Project goal: Match educational texts to correct achievement standards

### 2. Approach Comparison

#### Bi-Encoder (Cosine Similarity)

- Encodes two texts separately then computes cosine similarity
- Advantages: Fast (pre-computable), scalable
- Disadvantages: No interaction, lower accuracy
- Model: `jhgan/ko-sroberta-multitask`

#### Cross-Encoder (Re-ranking)

- Encodes two texts together to directly compute relevance score
- Advantages: Accurate (attention-based interaction)
- Disadvantages: Slow (computes all pairs each time)
- Usage: Bi-encoder extracts candidates → Cross-encoder re-ranks
- Model: `bongsoo/albert-small-kor-cross-encoder-v1`

#### LLM (Generative)

- Generative LLM directly outputs achievement standard code
- Advantages: Flexible, human-like reasoning, significant fine-tuning benefits
- Disadvantages: Slow, resource-intensive, prompt length limitations
- Models: `Qwen/Qwen2.5-3B-Instruct` (baseline), `unsloth/Qwen2.5-3B-Instruct` (fine-tuning)

#### RAG LLM (Retrieval-Augmented Generation)

- Two-stage approach: Multi-class classifier retrieves candidates, then LLM selects best match
- Advantages: Combines retrieval efficiency with LLM reasoning, handles large candidate spaces
- Disadvantages: Requires trained classifier, two-stage processing overhead
- Workflow: `infer_top_k` (retrieval) → LLM selection (generation)

### 3. Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)

- Bi-Encoder: Extracts candidates from entire space (recall-focused)
- Cross-Encoder: Precisely re-ranks candidates (precision-focused)
- Balance between speed and accuracy

### 4. RAG Workflow Architecture

- **Retrieval Stage**: Dense retrieval using trained multi-class classifier
  - Uses `infer_top_k` function from `src/classification/inference.py`
  - Retrieves top-k candidates ranked by classifier confidence
- **Generation Stage**: LLM selects best match from retrieved candidates
  - Reduces prompt length compared to including all candidates
  - Improves accuracy by focusing on relevant candidates

### 5. Evaluation Metrics

#### Top-K Accuracy

- Proportion of samples where ground truth is in top-k predictions
- K=1: Evaluates only highest prediction
- Higher K provides more lenient evaluation

#### MRR (Mean Reciprocal Rank)

- Mean reciprocal rank of ground truth
- Higher when ground truth appears in top ranks
- Range: 0~1 (1 is perfect)
- Calculation: RR = 1 / rank, MRR = average(RR)

#### Exact Match Percentage

- Proportion of samples where LLM generates exactly valid achievement standard code
- Distinguished from partial match

#### Match Type Distribution

- **exact**: Exactly matching code
- **partial**: Partial match (e.g., `03-04` → `10영03-04`)
- **invalid**: Invalid code

---
