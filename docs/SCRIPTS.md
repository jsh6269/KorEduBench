# Shell Scripts Documentation

이 문서는 `scripts/` 폴더에 있는 Shell 스크립트들의 실행 방법과 사용법을 설명합니다.

## 목차

### 데이터 전처리
1. [preprocess.sh](#1-preprocesssh) - 데이터 전처리

### 임베딩 기반 접근법
2. [cosine_similarity.sh](#2-cosine_similaritysh) - 코사인 유사도 평가
3. [cross_encoder.sh](#3-cross_encodersh) - Cross Encoder 학습 및 평가

### 멀티클래스 분류
4. [train_classifier.sh](#4-train_classifiersh) - 기본 분류기 학습
5. [train_advanced.sh](#5-train_advancedsh) - 고급 분류기
6. [eval_classifier.sh](#6-eval_classifiersh) - 분류기 평가

### LLM 기반 텍스트 분류
7. [llm_text_classification.sh](#7-llm_text_classificationsh) - LLM 평가
8. [finetuning_llm.sh](#8-finetuning_llmsh) - LLM 파인튜닝
9. [finetune_llm_text_classification.sh](#9-finetune_llm_text_classificationsh) - 파인튜닝된 LLM 평가

---

## 1. preprocess.sh

### 개요
AI Hub의 교육과정 수준 과목별 데이터셋을 전처리하는 스크립트입니다. 성취기준 추출, 텍스트 샘플 추가, train/validation 분할, few-shot 예시 생성을 수행합니다.

### 사전 요구사항
- [교육과정 수준 과목별 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM010&aihubDataSe=data&dataSetSn=71855) 다운로드 필요
- 데이터셋의 `label` 디렉토리 경로 확인 필요

### 주요 설정 (스크립트 내부 수정 필요)

```bash
BASE_DIR="/mnt/e/2025_2_KorEduBench"  # 데이터셋이 있는 기본 경로
MAX_TEXTS=80                           # 각 성취기준당 최대 텍스트 샘플 수
```

### 실행 방법

```bash
cd scripts
bash preprocess.sh
```


### 처리 단계

1. **Step 1: 성취기준 추출**
   - `extract_standards.py` 실행
   - 입력: `{BASE_DIR}/label/`
   - 출력: `dataset/unique_achievement_standards.csv`

2. **Step 2: 텍스트 샘플 추가**
   - `add_text_to_standards.py` 실행
   - 각 성취기준에 최대 `MAX_TEXTS`개의 텍스트 샘플 추가
   - 출력: `dataset/text_achievement_standards.csv`

3. **Step 3: Train/Validation 분할 및 과목별 분할**
   - `split_subject.py` 실행
   - Train/Validation 분할 (80/20)
   - 과목별로 CSV 파일 생성
   - Few-shot 예시 JSON 파일 생성
   - 출력:
     - `dataset/train.csv`, `dataset/valid.csv`
     - `dataset/train_80/{subject}.csv`, `dataset/valid_80/{subject}.csv`
     - `dataset/few_shot_examples/{subject}.json`
     - `dataset/insufficient_text.csv`

### 출력 파일

```
dataset/
├── unique_achievement_standards.csv
├── text_achievement_standards.csv
├── train.csv                  # 전체 train 데이터
├── valid.csv                  # 전체 validation 데이터
├── train_80/                  # 과목별 train (80 texts/standard)
│   ├── 과학.csv
│   ├── 국어.csv
│   ├── 수학.csv
│   ├── 영어.csv
│   ├── 사회.csv
│   ├── 사회문화.csv
│   ├── 도덕.csv
│   ├── 기술가정.csv
│   └── 정보.csv
├── valid_80/                  # 과목별 validation
│   └── ...
├── few_shot_examples/         # LLM few-shot 예시
│   ├── 과학.json
│   ├── 국어.json
│   └── ...
└── insufficient_text.csv      # 텍스트 샘플이 부족한 성취기준
```

### 커스터마이징

스크립트 파일을 열어 다음 변수들을 수정할 수 있습니다:

```bash
BASE_DIR="/your/dataset/path"  # 데이터셋 경로 변경
MAX_TEXTS=80                   # 텍스트 샘플 수 (기본값: 80)
```

---

## 2. cosine_similarity.sh

### 개요
jhgan/ko-sroberta-multitask 기반
과목별 CSV 파일에 대해 코사인 유사도 기반 평가를 수행하는 스크립트입니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `dataset/valid_80/` 디렉토리 존재

### 주요 설정

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"  # 평가할 데이터셋 폴더
```

### 실행 방법

```bash
cd scripts
bash cosine_similarity.sh
```


### 동작 방식

1. `valid_80` 폴더 내 모든 CSV 파일을 찾습니다 (9개 과목)
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

### 커스터마이징

다른 데이터셋 폴더를 평가하려면:

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/train_80"  # Train 데이터로 평가
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
TRAIN_CSV="${PROJECT_ROOT}/dataset/train_80/과학.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/valid_80/과학.csv"
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

### 커스터마이징

다른 과목으로 학습/평가하려면:

```bash
TRAIN_CSV="${PROJECT_ROOT}/dataset/train_80/국어.csv"
VALIDATION_CSV="${PROJECT_ROOT}/dataset/valid_80/국어.csv"
```

---

## 4. train_classifier.sh

### 개요
기본 멀티클래스 분류기를 학습하는 스크립트입니다. Transformer 기반 모델을 사용하여 성취기준 분류를 수행합니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `dataset/train_80/` 디렉토리 존재

### 실행 방법

```bash
cd scripts
bash train_classifier.sh
```

### 주요 특징
- Transformer 기반 분류 모델 학습
- 기본 Cross-Entropy Loss 사용
- 표준 학습 설정 적용

### 출력
- 학습된 분류 모델
- 학습 로그 및 평가 메트릭

---

## 5. train_advanced.sh

### 개요
고급 학습 기법을 적용한 분류기 학습 스크립트입니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료

### 실행 방법

```bash
cd scripts
bash train_advanced.sh
```

### 주요 특징
- 고급 정규화 기법
- 최적화된 하이퍼파라미터
- 향상된 데이터 증강

---

## 6. eval_classifier.sh

### 개요
분류기 평가 스크립트입니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `train_advanced.sh` 실행 완료

### 실행 방법

```bash
cd scripts
bash eval_classifier.sh
```

---

## 7. llm_text_classification.sh

### 개요
LLM(Large Language Model)을 사용한 텍스트 분류 평가 스크립트입니다. Validation 데이터셋의 모든 과목을 순차적으로 처리합니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- CUDA 지원 GPU (권장)
- 충분한 VRAM (모델 크기에 따라 다름)

### 주요 설정

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # 사용할 LLM 모델
MAX_NEW_TOKENS=50                       # 생성할 최대 토큰 수
TEMPERATURE=0.1                         # 샘플링 온도 (낮을수록 결정적)
DEVICE="cuda"                           # 디바이스 (cuda 또는 cpu)
MAX_INPUT_LENGTH=6144                   # 최대 입력 길이
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

1. `valid_80` 폴더 내 모든 CSV 파일을 찾습니다 (9개 과목)
2. 각 CSV 파일(과목)에 대해 순차적으로:
   - 해당 과목의 few-shot 예시를 로드합니다 (`few_shot_examples/{subject}.json`)
   - LLM을 로드합니다
   - Few-shot 프롬프팅으로 성취기준을 예측합니다
   - 정확도와 MRR을 계산합니다
   - 정답/오답 샘플을 저장합니다
3. 에러가 발생해도 다음 파일을 계속 처리합니다

### 출력 파일

```
output/
└── llm_text_classification/
    ├── results.json           # 모든 과목의 평가 결과
    └── logs/
        ├── 과학_corrects.txt   # 과목별 정답 샘플 로그
        ├── 과학_wrongs.txt     # 과목별 오답 샘플 로그
        ├── 국어_corrects.txt
        ├── 국어_wrongs.txt
        └── ...
```

**results.json 예시:**
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
    "max_samples_per_row": 80,
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
DATASET_FOLDER="${PROJECT_ROOT}/dataset/train_80"
```

---

## 8. finetuning_llm.sh

### 개요
LLM을 성취기준 분류 태스크에 맞게 파인튜닝하는 스크립트입니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `dataset/train_80/` 디렉토리 존재
- 충분한 VRAM (LLM 파인튜닝용)

### 실행 방법

```bash
cd scripts
bash finetuning_llm.sh
```

### 주요 설정

```bash
MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # 파인튜닝할 기본 모델
OUTPUT_DIR="${PROJECT_ROOT}/model/finetuned_llm"   # 파인튜닝된 모델 저장 경로
```

### 동작 방식

1. Train 데이터셋을 로드합니다
2. 기본 LLM을 로드합니다
3. 성취기준 분류 태스크에 맞게 파인튜닝합니다
4. 파인튜닝된 모델을 저장합니다

### 출력 파일

```
model/
└── finetuned_llm/           # 파인튜닝된 LLM
    ├── config.json
    ├── model weights
    ├── tokenizer files
    └── training_args.json
```

### 주요 특징
- Instruction tuning 방식 적용
- 효율적인 파인튜닝 기법 (LoRA, QLoRA 등) 사용 가능
- 학습 로그 및 체크포인트 저장

---

## 9. finetune_llm_text_classification.sh

### 개요
파인튜닝된 LLM을 평가하는 스크립트입니다. Validation 데이터셋의 모든 과목을 순차적으로 처리합니다.

### 사전 요구사항
- `preprocess.sh` 실행 완료
- `finetuning_llm.sh` 실행 완료 (파인튜닝된 모델 필요)
- `model/finetuned_llm/` 디렉토리 존재

### 주요 설정

```bash
DATASET_FOLDER="${PROJECT_ROOT}/dataset/valid_80"
MODEL_PATH="${PROJECT_ROOT}/model/finetuned_llm"  # 파인튜닝된 모델 경로
MAX_NEW_TOKENS=50
TEMPERATURE=0.1
DEVICE="cuda"
MAX_INPUT_LENGTH=6144
MAX_TOTAL_SAMPLES=100
```

### 실행 방법

```bash
cd scripts
bash finetune_llm_text_classification.sh
```

### 동작 방식

1. 파인튜닝된 LLM을 로드합니다
2. `valid_80` 폴더 내 모든 CSV 파일을 처리합니다 (9개 과목)
3. 각 과목에 대해:
   - Few-shot 예시와 함께 평가
   - 정답/오답 샘플 저장
   - 성능 메트릭 계산

### 출력 파일

```
output/
└── llm_text_classification/
    ├── finetuned_results.json    # 파인튜닝된 LLM 평가 결과
    └── finetuned_logs/
        ├── finetuned_llm_과학_corrects.txt
        ├── finetuned_llm_과학_wrongs.txt
        ├── finetuned_llm_국어_corrects.txt
        ├── finetuned_llm_국어_wrongs.txt
        └── ...
```

**finetuned_results.json 예시:**
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
    "max_samples_per_row": 80,
    "total_samples": 100,
    "correct": 75,
    "accuracy": 0.75,
    "mrr": 0.82,
    "exact_match_count": 70,
    "exact_match_percentage": 0.70,
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

### 주요 특징
- 파인튜닝 효과 측정
- 사전 학습 모델과 성능 비교 가능
- 상세한 로그 및 분석 자료 제공

---

## 실행 순서 (권장)

전체 파이프라인을 처음부터 실행하는 경우:

### 기본 파이프라인

```bash
cd scripts

# 1단계: 데이터 전처리
bash preprocess.sh

# 2단계: 코사인 유사도 베이스라인 평가
bash cosine_similarity.sh

# 3단계: Cross Encoder 학습 및 평가
bash cross_encoder.sh

# 4단계: LLM 기반 분류 평가
bash llm_text_classification.sh
```

### 분류기 학습 파이프라인

```bash
cd scripts

# 데이터 전처리 (이미 완료했다면 스킵)
bash preprocess.sh

# 다양한 분류기 학습
bash train_classifier.sh              # 기본 분류기
bash train_advanced.sh                # 고급 분류기
```

### LLM 파인튜닝 파이프라인

```bash
cd scripts

# 데이터 전처리 (이미 완료했다면 스킵)
bash preprocess.sh

# LLM 파인튜닝 및 평가
bash finetuning_llm.sh                        # LLM 파인튜닝
bash finetune_llm_text_classification.sh      # 파인튜닝된 LLM 평가

# 비교를 위한 사전학습 LLM 평가
bash llm_text_classification.sh               # 사전학습 LLM 평가
```

## 공통 사항

### 경로 설정
- 모든 스크립트는 프로젝트 루트를 자동으로 찾습니다
- 어느 디렉토리에서든 실행 가능합니다

### 출력 형식
- 모든 스크립트는 컬러 터미널 출력을 지원합니다
- 진행 상황과 결과를 실시간으로 표시합니다

### 에러 처리
- **즉시 중단 방식**: `preprocess.sh`, `cosine_similarity.sh`, `cross_encoder.sh`, 분류기 학습 스크립트들
  - 에러 발생 시 즉시 중단 (`set -e`)
- **계속 진행 방식**: `llm_text_classification.sh`, `finetune_llm_text_classification.sh`
  - 에러 발생 시에도 다음 파일 계속 처리 (배치 처리)

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
→ Or `dataset/dataset_bundle.tar.gz`를 이용해서 sample dataset 생성

---
