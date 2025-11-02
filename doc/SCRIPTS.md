# Shell Scripts Documentation

이 문서는 `scripts/` 폴더에 있는 4개의 Shell 스크립트의 실행 방법과 사용법을 설명합니다.

## 목차

1. [preprocess.sh](#1-preprocesssh) - 데이터 전처리
2. [cosine_similarity.sh](#2-cosine_similaritysh) - 코사인 유사도 평가
3. [cross_encoder.sh](#3-cross_encodersh) - Cross Encoder 학습 및 평가
4. [llm_text_classification.sh](#4-llm_text_classificationsh) - LLM 기반 텍스트 분류

---

## 1. preprocess.sh

### 개요
AI Hub의 교육과정 수준 과목별 데이터셋을 전처리하는 스크립트입니다. Training과 Validation 데이터셋 모두를 처리합니다.

### 사전 요구사항
- [교육과정 수준 과목별 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71855) 다운로드 필요
- 데이터셋의 `label` 디렉토리 경로 확인 필요

### 주요 설정 (스크립트 내부 수정 필요)

```bash
BASE_DIR="/mnt/e/2025_2_KorEduBench"  # 데이터셋이 있는 기본 경로
MAX_TEXTS=20                           # 각 성취기준당 최대 텍스트 샘플 수
```

### 실행 방법

```bash
cd scripts
bash preprocess.sh
```


### 처리 단계

스크립트는 각 데이터셋(Training, Validation)에 대해 다음 3단계를 수행합니다:

1. **Step 1: 성취기준 추출**
   - `extract_standards.py` 실행
   - 입력: `{BASE_DIR}/{Training|Validation}/label/`
   - 출력: `dataset/{prefix}_unique_achievement_standards.csv`

2. **Step 2: 텍스트 샘플 추가**
   - `add_text_to_standards.py` 실행
   - 각 성취기준에 최대 `MAX_TEXTS`개의 텍스트 샘플 추가
   - 출력: `dataset/{prefix}_text_achievement_standards.csv`

3. **Step 3: 과목별 분할**
   - `split_subject.py` 실행
   - 과목별로 CSV 파일 생성
   - 출력: `dataset/{prefix}_subject_text{MAX_TEXTS}/`

### 출력 파일

```
dataset/
├── training_unique_achievement_standards.csv
├── training_text_achievement_standards.csv
├── training_subject_text20/
│   ├── 과학.csv
│   ├── 국어.csv
│   ├── 수학.csv
│   ├── 사회.csv
│   └── ...
├── validation_unique_achievement_standards.csv
├── validation_text_achievement_standards.csv
└── validation_subject_text20/
    ├── 과학.csv
    ├── 국어.csv
    ├── 수학.csv
    └── ...
```

### 커스터마이징

스크립트 파일을 열어 다음 변수들을 수정할 수 있습니다:

```bash
BASE_DIR="/your/dataset/path"  # 데이터셋 경로 변경
MAX_TEXTS=30                   # 텍스트 샘플 수 변경
```

---

## 2. cosine_similarity.sh

### 개요
과목별 CSV 파일에 대해 코사인 유사도 기반 평가를 수행하는 스크립트입니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `dataset/training_subject_text20/` 디렉토리 존재

### 주요 설정

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/training_subject_text20"  # 평가할 데이터셋 폴더
```

### 실행 방법

```bash
cd scripts
bash cosine_similarity.sh
```


### 동작 방식

1. `training_subject_text20` 폴더 내 모든 CSV 파일을 찾습니다
2. 각 CSV 파일에 대해 코사인 유사도 평가를 수행합니다
3. SentenceTransformer 모델을 사용하여 텍스트와 성취기준을 임베딩합니다
4. Top-k 정확도와 MRR(Mean Reciprocal Rank)을 계산합니다

### 출력 파일

```
output/
└── cosine_similarity/
    └── results.json  # 평가 결과가 JSON 형식으로 저장됨
```

**results.json 예시:**
```json
[
  {
    "folder": "training",
    "model_name": "jhgan/ko-sroberta-multitask",
    "subject": "과학",
    "num_standards": 150,
    "total_samples": 3000,
    "top1_acc": 0.7234,
    "top3_acc": 0.8456,
    "top10_acc": 0.9123,
    "mrr": 0.7891
  }
]
```

### 커스터마이징

다른 데이터셋 폴더를 평가하려면:

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/validation_subject_text20"
```

---

## 3. cross_encoder.sh

### 개요
Cross Encoder 모델을 fine-tuning하고 평가하는 스크립트입니다. 과학 과목을 기본으로 학습하고 평가합니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- Training 및 Validation 데이터셋 준비 완료

### 주요 설정

```bash
TRAIN_CSV="${PROJECT_ROOT}/dataset/training_subject_text20/과학.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/validation_subject_text20/과학.csv"
```

### 실행 방법

```bash
cd scripts
bash cross_encoder.sh
```

또는

```bash
cd /home/jeongmin/projects/2025_nlp/KorEduBench
bash scripts/cross_encoder.sh
```

### 처리 단계

1. **Step 1: Cross Encoder Fine-tuning**
   - `finetune_cross_encoder.py` 실행
   - Training 데이터로 모델 학습
   - 출력: `model/cross_finetuned/`

2. **Step 2: Cross Encoder Evaluation**
   - `eval_cross_encoder.py` 실행
   - Bi-encoder로 후보를 검색한 후 Cross Encoder로 재정렬
   - Validation 데이터로 평가
   - 출력: `output/cross_encoder/results_rerank.json`

### 출력 파일

```
model/
└── cross_finetuned/           # Fine-tuned Cross Encoder 모델
    ├── config.json
    ├── pytorch_model.bin
    └── ...

output/
└── cross_encoder/
    ├── results_rerank.json    # 평가 결과
    └── logs/
        └── 과학_wrongs.txt     # 오분류된 샘플 로그 (최대 100개)
```

**results_rerank.json 예시:**
```json
[
  {
    "folder": "validation",
    "bi_model": "jhgan/ko-sroberta-multitask",
    "cross_model": "model/cross_finetuned",
    "subject": "과학",
    "top_k": 20,
    "top1_acc": 0.8234,
    "top3_acc": 0.9156,
    "mrr": 0.8591
  }
]
```

### 커스터마이징

다른 과목으로 학습/평가하려면:

```bash
TRAIN_CSV="${PROJECT_ROOT}/dataset/training_subject_text20/국어.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/validation_subject_text20/국어.csv"
```

---

## 4. llm_text_classification.sh

### 개요
LLM(Large Language Model)을 사용한 텍스트 분류 평가 스크립트입니다. Validation 데이터셋의 모든 과목을 순차적으로 처리합니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- CUDA 지원 GPU (권장)
- 충분한 VRAM (모델 크기에 따라 다름)

### 주요 설정

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/validation_subject_text20"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # 사용할 LLM 모델
MAX_NEW_TOKENS=200                      # 생성할 최대 토큰 수
TEMPERATURE=0.1                         # 샘플링 온도 (낮을수록 결정적)
DEVICE="cuda"                           # 디바이스 (cuda 또는 cpu)
MAX_INPUT_LENGTH=2048                   # 최대 입력 길이
MAX_TOTAL_SAMPLES=100                   # 평가할 최대 샘플 수 (None이면 전체)
```

### 실행 방법

```bash
cd scripts
bash llm_text_classification.sh
```

또는

```bash
cd /home/jeongmin/projects/2025_nlp/KorEduBench
bash scripts/llm_text_classification.sh
```

### 동작 방식

1. `validation_subject_text20` 폴더 내 모든 CSV 파일을 찾습니다
2. 각 CSV 파일(과목)에 대해 순차적으로:
   - LLM을 로드합니다
   - 텍스트를 입력으로 성취기준을 예측합니다
   - Top-k 정확도와 MRR을 계산합니다
   - 결과와 오분류 샘플을 저장합니다
3. 에러가 발생해도 다음 파일을 계속 처리합니다

### 출력 파일

```
output/
└── llm_text_classification/
    ├── results.json           # 모든 과목의 평가 결과
    └── logs/
        ├── 과학_wrongs.txt     # 과목별 오분류 샘플 로그
        ├── 국어_wrongs.txt
        ├── 수학_wrongs.txt
        └── ...
```

**results.json 예시:**
```json
[
  {
    "folder": "validation",
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "subject": "과학",
    "num_standards": 150,
    "total_samples": 100,
    "top1_acc": 0.65,
    "top3_acc": 0.82,
    "top10_acc": 0.91,
    "mrr": 0.7234
  }
]
```

### 커스터마이징

#### 다른 LLM 모델 사용

```bash
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
```

#### CPU에서 실행

```bash
DEVICE="cpu"
```

#### 전체 데이터셋 평가

```bash
MAX_TOTAL_SAMPLES=None  # 또는 매우 큰 숫자
```

#### Training 데이터셋 평가

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/training_subject_text20"
```

---

## 실행 순서 (권장)

전체 파이프라인을 처음부터 실행하는 경우:

```bash
cd /home/jeongmin/projects/2025_nlp/KorEduBench/scripts

# 1단계: 데이터 전처리
bash preprocess.sh

# 2단계: 코사인 유사도 베이스라인 평가
bash cosine_similarity.sh

# 3단계: Cross Encoder 학습 및 평가
bash cross_encoder.sh

# 4단계: LLM 기반 분류 평가
bash llm_text_classification.sh
```

## 공통 사항

### 경로 설정
- 모든 스크립트는 프로젝트 루트를 자동으로 찾습니다
- 어느 디렉토리에서든 실행 가능합니다

### 출력 형식
- 모든 스크립트는 컬러 터미널 출력을 지원합니다
- 진행 상황과 결과를 실시간으로 표시합니다

### 에러 처리
- `preprocess.sh`, `cosine_similarity.sh`, `cross_encoder.sh`: 에러 발생 시 즉시 중단 (`set -e`)
- `llm_text_classification.sh`: 에러 발생 시에도 다음 파일 계속 처리

### 스크립트 수정
모든 스크립트는 텍스트 에디터로 열어 설정을 수정할 수 있습니다:

```bash
# 예시
vim scripts/llm_text_classification.sh
nano scripts/cross_encoder.sh
```

## 문제 해결

### 데이터셋을 찾을 수 없음
```
Error: Dataset folder not found
```
→ `preprocess.sh` 먼저 실행하여 데이터셋 생성

### CUDA 메모리 부족
```
CUDA out of memory
```
→ `llm_text_classification.sh`에서:
- `MODEL_NAME`을 더 작은 모델로 변경
- `MAX_INPUT_LENGTH` 줄이기
- `DEVICE="cpu"` 사용 (느림)

### 모델 다운로드 실패
```
Failed to load model
```
→ 인터넷 연결 확인 및 Hugging Face 접근 권한 확인

---


