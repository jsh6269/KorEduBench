# KorEduBench 코드 상세 분석

이 문서는 KorEduBench 프로젝트의 각 코드 파일을 상세히 분석하여, 각 파일이 수행하는 작업을 단계별로 설명합니다.

---

## 목차

1. [데이터 전처리 파이프라인](#1-데이터-전처리-파이프라인)
   - 1.1 [extract_standards.py](#11-extract_standardspy)
   - 1.2 [add_text_to_standards.py](#12-add_text_to_standardspy)
   - 1.3 [split_subject.py](#13-split_subjectpy)

2. [Cosine Similarity 평가](#2-cosine-similarity-평가)
   - 2.1 [eval_cosine_similarity.py](#21-eval_cosine_similaritypy)
   - 2.2 [batch_cosine_similarity.py](#22-batch_cosine_similaritypy)

3. [Cross Encoder 학습 및 평가](#3-cross-encoder-학습-및-평가)
   - 3.1 [finetune_cross_encoder.py](#31-finetune_cross_encoderpy)
   - 3.2 [eval_cross_encoder.py](#32-eval_cross_encoderpy)

---

## 1. 데이터 전처리 파이프라인

### 1.1 extract_standards.py

**위치**: `dataset/extract_standards.py`

**목적**: AI Hub 데이터셋의 ZIP 파일들로부터 2022 성취기준(achievement standard)을 추출하여 CSV로 저장

#### 주요 기능 분석

**함수**: `extract_unique_standards(label_dir, output_csv)`

**단계별 처리 과정**:

1. **ZIP 파일 수집** (줄 22-28)
   - `label_dir` 디렉토리를 순회하며 모든 `.zip` 파일 수집
   - `os.walk()`로 하위 디렉토리까지 재귀적으로 탐색

2. **ZIP 파일 처리** (줄 31-72)
   - 각 ZIP 파일 내부의 JSON 파일들을 순회
   - JSON에서 다음 정보 추출:
     - `source_data_info.2022_achievement_standard`: 성취기준 리스트
     - `raw_data_info`: subject(과목), school(학교급), grade(학년)
   
3. **성취기준 파싱** (줄 54-67)
   - 형식: `"[코드] 내용"` (예: `"[6과01-01] 물체의 운동을 관찰하여..."`)
   - `[` 와 `]` 사이의 텍스트를 코드로 추출
   - `]` 이후의 텍스트를 내용(content)으로 추출
   - 중복 제거: `unique_standards` 딕셔너리에 코드를 키로 사용

4. **CSV 저장** (줄 74-86)
   - 컬럼: `subject`, `school`, `grade`, `code`, `content`
   - 정렬 기준: 과목(subject) → 코드(code) 순
   - UTF-8-sig 인코딩으로 저장 (Excel 호환)

**입력**: 
- `label_dir`: ZIP 파일들이 있는 디렉토리 경로

**출력**: 
- `unique_achievement_standards.csv`: 고유한 성취기준 목록

**예시 실행**:
```bash
python extract_standards.py ./label
```

---

### 1.2 add_text_to_standards.py

**위치**: `dataset/add_text_to_standards.py`

**목적**: 추출된 성취기준 CSV에 실제 교육 텍스트 샘플들을 매칭하여 추가

#### 주요 기능 분석

**함수**: `append_texts_to_csv(label_dir, csv_path, output_csv, max_texts)`

**단계별 처리 과정**:

1. **CSV 로딩 및 준비** (줄 20-38)
   - `chardet`로 CSV 인코딩 자동 감지
   - pandas DataFrame으로 CSV 로딩
   - `text_1`, `text_2`, ..., `text_N` 컬럼을 `max_texts` 개수만큼 생성
   - 기존에 `text_` 컬럼이 있다면 이어서 추가

2. **코드-인덱스 매핑 생성** (줄 39)
   - `code_to_idx`: 성취기준 코드를 DataFrame 행 인덱스로 매핑
   - 빠른 검색을 위한 해시맵 구조

3. **ZIP 파일 순회 및 텍스트 추출** (줄 41-93)
   - 각 JSON 파일에서 학습 데이터 추출:
     - `learning_data_info.text_description`: 텍스트 설명
     - `learning_data_info.text_qa`: 질문-답변 텍스트
     - `learning_data_info.text_an`: 추가 텍스트
   - 세 가지 텍스트를 공백으로 결합하여 `combined_text` 생성

4. **텍스트-성취기준 매칭** (줄 76-88)
   - JSON의 `2022_achievement_standard`에서 코드 추출
   - 해당 코드가 CSV에 있으면 빈 `text_` 컬럼에 텍스트 추가
   - 이미 `max_texts`만큼 채워진 경우 건너뜀

5. **결과 저장** (줄 96-97)
   - 업데이트된 DataFrame을 CSV로 저장
   - UTF-8-sig 인코딩 사용

**입력**: 
- `label_dir`: ZIP 파일 디렉토리
- `csv_path`: 입력 CSV (성취기준만 있는 파일)
- `max_texts`: 추가할 최대 텍스트 개수 (기본값: 10)

**출력**: 
- `text_achievement_standards.csv`: 텍스트 샘플이 추가된 CSV

**예시 실행**:
```bash
python add_text_to_standards.py ./label --csv_path unique_achievement_standards.csv --max_texts 20
```

---

### 1.3 split_subject.py

**위치**: `dataset/split_subject.py`

**목적**: 통합된 CSV 파일을 과목별로 분할하여 개별 파일로 저장

#### 주요 기능 분석

**함수**: `split_csv_by_subject(input_path, max_texts, encoding)`

**단계별 처리 과정**:

1. **출력 디렉토리 생성** (줄 11-12)
   - 형식: `subject_text{max_texts}` (예: `subject_text20`)
   - 이미 존재해도 오류 없음 (`exist_ok=True`)

2. **CSV 로딩 및 컬럼 선택** (줄 13-23)
   - 기본 컬럼: `subject`, `school`, `grade`, `code`, `content`
   - 텍스트 컬럼: `text_1` ~ `text_{max_texts}`
   - 존재하는 컬럼만 선택 (일부만 있어도 동작)

3. **과목별 그룹화 및 저장** (줄 26-35)
   - `df.groupby("subject")`로 과목별로 데이터 분할
   - 파일명 안전화: 영숫자와 `_`만 허용, 공백은 `_`로 변환
   - 각 과목을 `{과목명}.csv`로 저장
   - 예시: `과학.csv`, `수학.csv`, `영어.csv`

**입력**: 
- `input_path`: 통합 CSV 파일 경로
- `max_texts`: 포함할 텍스트 컬럼 개수
- `encoding`: CSV 인코딩 (기본값: utf-8)

**출력**: 
- `subject_text{N}/` 디렉토리 내 과목별 CSV 파일들

**예시 실행**:
```bash
python split_subject.py --input text_achievement_standards.csv --max-texts 20
```

---

## 2. Cosine Similarity 평가

### 2.1 eval_cosine_similarity.py

**위치**: `cosine_similarity/eval_cosine_similarity.py`

**목적**: Bi-Encoder를 사용하여 코사인 유사도 기반 검색 성능 평가 (베이스라인)

#### 주요 기능 분석

**함수**: `evaluate_cosine_similarity_baseline(...)`

**단계별 처리 과정**:

1. **데이터 준비** (줄 42-69)
   - CSV 인코딩 자동 감지 (`detect_encoding`)
   - 필수 컬럼 확인: `code`, `content`
   - `text_` 컬럼들 찾기 (평가용 샘플)
   - `max_samples_per_row` 자동 계산 (각 행의 최대 텍스트 개수)

2. **Bi-Encoder 모델 로딩** (줄 71-74)
   - `SentenceTransformer` 모델 로딩
   - 기본 모델: `jhgan/ko-sroberta-multitask` (한국어 문장 임베딩)
   - 평가 모드로 설정

3. **성취기준 인코딩** (줄 76-82)
   - 모든 성취기준의 `content` 필드를 벡터로 변환
   - GPU 텐서로 변환 (`convert_to_tensor=True`)
   - 진행 상황 표시 (`show_progress_bar=True`)

4. **샘플 텍스트 수집 및 인코딩** (줄 84-109)
   - 각 행의 `text_1`, `text_2`, ... 컬럼에서 텍스트 추출
   - `max_samples_per_row` 제한 적용
   - 샘플과 정답 코드 쌍 생성
   - 모든 샘플 텍스트를 벡터로 인코딩

5. **코사인 유사도 계산** (줄 111-113)
   - `util.cos_sim(emb_samples, emb_contents)`: 유사도 행렬 계산
   - 크기: `[num_samples, num_standards]`
   - 각 샘플이 모든 성취기준과 얼마나 유사한지 계산

6. **Top-K 정확도 계산** (줄 115-124)
   - Top-1, 3, 10, 20, 40, 60 정확도 계산
   - 각 샘플에 대해 상위 k개 예측 중 정답이 있는지 확인
   - 정확도 = 맞춘 샘플 수 / 전체 샘플 수

7. **MRR (Mean Reciprocal Rank) 계산** (줄 126-137)
   - 각 샘플의 정답이 몇 번째 순위인지 확인
   - Reciprocal Rank = 1 / 순위
   - MRR = 모든 샘플의 RR 평균
   - 높을수록 정답이 상위에 랭크됨을 의미

8. **결과 저장** (줄 139-202)
   - JSON 파일로 결과 저장 (`results.json`)
   - 기존 결과가 있으면 업데이트, 없으면 추가
   - 저장 정보: 모델명, 과목, 정확도, MRR 등

**평가 메트릭**:
- **Top-K Accuracy**: 상위 K개 예측 중 정답 포함 비율
- **MRR**: 정답의 평균 역순위 (1위면 1.0, 2위면 0.5, 3위면 0.33...)

**입력**: 
- `input_csv`: 평가할 CSV 파일
- `model_name`: Bi-Encoder 모델 이름
- `max_samples_per_row`: 행당 최대 샘플 수

**출력**: 
- `results.json`: 평가 결과

**예시 실행**:
```bash
python eval_cosine_similarity.py --input_csv ../dataset/subject_text20/과학.csv
```

---

### 2.2 batch_cosine_similarity.py

**위치**: `cosine_similarity/batch_cosine_similarity.py`

**목적**: 폴더 내 모든 CSV 파일에 대해 코사인 유사도 평가를 일괄 실행

#### 주요 기능 분석

**함수**: `evaluate_folder(...)`

**단계별 처리 과정**:

1. **CSV 파일 수집** (줄 16-24)
   - `folder_path` 내 모든 `.csv` 파일 찾기
   - 파일이 없으면 경고 메시지 출력

2. **각 CSV 파일 평가** (줄 29-40)
   - `tqdm`으로 진행 상황 표시
   - 각 파일에 대해 `evaluate_cosine_similarity_baseline` 호출
   - 오류 발생 시 건너뛰고 계속 진행
   - 모든 결과를 동일한 JSON 파일에 누적 저장

**입력**: 
- `folder_path`: CSV 파일들이 있는 폴더
- `model_name`: 사용할 Bi-Encoder 모델
- `json_path`: 결과를 저장할 JSON 파일

**출력**: 
- `results.json`: 모든 과목의 평가 결과

**예시 실행**:
```bash
python batch_cosine_similarity.py --folder_path ../dataset/subject_text20/
```

---

## 3. Cross Encoder 학습 및 평가

### 3.1 finetune_cross_encoder.py

**위치**: `cross_encoder/finetune_cross_encoder.py`

**목적**: Cross-Encoder 모델을 한국어 교육 데이터에 파인튜닝

#### 주요 기능 분석

**함수**: `fine_tune_cross_encoder(...)`

#### Cross-Encoder란?
- Bi-Encoder: 텍스트를 별도로 인코딩 후 유사도 계산 (빠르지만 정확도 낮음)
- Cross-Encoder: 두 텍스트를 함께 입력하여 직접 관련성 점수 계산 (느리지만 정확)

**단계별 처리 과정**:

1. **데이터 로딩 및 분할** (줄 94-103)
   - CSV에서 성취기준 데이터 로딩
   - Train/Test 분할 (기본 8:2)
   - 행(row) 단위로 분할하여 데이터 누수 방지

2. **학습 데이터 쌍 생성** (`build_pairs_from_df`, 줄 34-80)
   
   **긍정 쌍(Positive Pairs) 생성** (줄 40-54):
   - 각 성취기준(`content`)과 해당 텍스트 샘플들 매칭
   - 형식: `(text_sample, achievement_standard_content)` → label=1.0
   - 중복 제거를 위해 `pos_keys` 집합 사용
   
   **부정 쌍(Negative Pairs) 생성** (줄 56-77):
   - 랜덤하게 텍스트와 성취기준을 매칭 (관련 없는 쌍)
   - 형식: `(text_sample, random_content)` → label=0.0
   - 긍정 쌍과 겹치지 않도록 검증
   - 부정 쌍 개수 = 긍정 쌍 개수 × `neg_ratio`
   - 최대 시도 횟수: `num_neg * 20`

3. **모델 초기화** (줄 113)
   - 기본 모델: `bongsoo/albert-small-kor-cross-encoder-v1`
   - 한국어에 특화된 경량 Cross-Encoder
   - `num_labels=1`: 연속적인 관련성 점수 출력 (0~1)

4. **학습 설정** (줄 114-117)
   - DataLoader 생성 (배치 처리)
   - Warmup steps: 전체 스텝의 10%
   - Learning rate 서서히 증가 후 감소 (학습 안정화)

5. **파인튜닝 실행** (줄 120-128)
   - `model.fit()` 호출
   - 에폭 수, Learning rate, Warmup 설정
   - 진행 상황 표시
   - 학습된 모델을 `output_dir`에 저장

6. **평가** (줄 131-145)
   - Test set에서 예측 수행
   - 0.5를 기준으로 이진 분류
   - 메트릭 계산:
     - **Accuracy**: 정확히 분류한 비율
     - **F1 Score**: Precision과 Recall의 조화평균
     - **ROC-AUC**: 분류 성능의 종합 지표 (0.5=랜덤, 1.0=완벽)

**입력**: 
- `input_csv`: 학습용 CSV 파일
- `base_model`: 사전학습된 Cross-Encoder 모델
- `epochs`: 학습 에폭 수 (기본값: 2)
- `batch_size`: 배치 크기 (기본값: 8)

**출력**: 
- `cross_finetuned/`: 파인튜닝된 모델 디렉토리
- 콘솔에 Accuracy, F1, ROC-AUC 출력

**예시 실행**:
```bash
python finetune_cross_encoder.py --input_csv ../dataset/subject_text20/과학.csv --epochs 3
```

---

### 3.2 eval_cross_encoder.py

**위치**: `cross_encoder/eval_cross_encoder.py`

**목적**: Bi-Encoder로 후보를 추출한 후 Cross-Encoder로 재순위화하여 정확도 향상

#### 주요 기능 분석

**함수**: `evaluate_bi_cross_pipeline(...)`

#### 2단계 검색 파이프라인
1. **Bi-Encoder**: 빠르게 상위 K개 후보 추출
2. **Cross-Encoder**: 후보들을 정밀하게 재순위화

**단계별 처리 과정**:

1. **데이터 및 모델 준비** (줄 44-86)
   - CSV 로딩 및 검증
   - Bi-Encoder 로딩 (`jhgan/ko-sroberta-multitask`)
   - Cross-Encoder 로딩 (파인튜닝된 모델)
   - 성취기준 벡터화 (Bi-Encoder)

2. **샘플 수집** (줄 88-104)
   - `text_` 컬럼에서 평가 샘플 추출
   - 정답 코드와 함께 저장
   - `max_samples_per_row` 제한 적용

3. **1단계: Bi-Encoder 검색** (줄 106-114)
   - 모든 샘플을 벡터로 인코딩
   - 코사인 유사도 계산
   - 각 샘플당 상위 `top_k`개 후보 선택 (기본값: 20)
   - 빠른 속도로 후보군 축소

4. **2단계: Cross-Encoder 재순위화** (줄 116-165)
   
   각 샘플에 대해:
   - `top_k`개 후보와 query를 쌍으로 만듦
   - Cross-Encoder로 정확한 관련성 점수 계산
   - 점수 기준으로 재정렬
   - Top-1 예측이 틀린 경우 오답 샘플로 기록

5. **평가 메트릭 계산** (줄 167-172)
   - **Top-1/3/10/20 Accuracy**: 상위 K개 내 정답 포함 비율
   - **MRR**: 평균 역순위
   - Bi-Encoder 단독보다 일반적으로 높은 성능

6. **결과 저장** (줄 174-259)
   
   **JSON 로깅** (줄 186-237):
   - 모델 정보, 과목, 정확도, MRR
   - 기존 결과 업데이트 또는 추가
   - `results_rerank.json`에 저장
   
   **오답 분석** (줄 239-259):
   - 틀린 샘플 중 랜덤 100개 저장
   - 저장 정보:
     - 입력 텍스트
     - 정답 코드 및 내용
     - 예측 코드 및 내용
   - `logs/{과목명}_wrongs.txt`에 저장

**입력**: 
- `input_csv`: 평가할 CSV 파일
- `bi_model`: Bi-Encoder 모델
- `cross_model`: Cross-Encoder 모델 (파인튜닝된 것)
- `top_k`: 재순위화할 후보 개수

**출력**: 
- `results_rerank.json`: 평가 결과
- `logs/{subject}_wrongs.txt`: 오답 샘플 분석

**예시 실행**:
```bash
python eval_cross_encoder.py --input_csv ../dataset/subject_text20/과학.csv --cross_model ./cross_finetuned
```

---

## 전체 파이프라인 요약

### Phase 1: 데이터 준비
```
원본 ZIP 파일
    ↓ extract_standards.py
unique_achievement_standards.csv (성취기준만)
    ↓ add_text_to_standards.py
text_achievement_standards.csv (텍스트 샘플 추가)
    ↓ split_subject.py
subject_text20/과학.csv, 수학.csv, ... (과목별 분할)
```

### Phase 2: 베이스라인 평가
```
과목별 CSV
    ↓ eval_cosine_similarity.py (또는 batch)
results.json (Bi-Encoder 단독 성능)
```

### Phase 3: Cross-Encoder 파인튜닝
```
과목별 CSV
    ↓ finetune_cross_encoder.py
cross_finetuned/ (파인튜닝된 모델)
```

### Phase 4: 재순위화 평가
```
과목별 CSV + 파인튜닝된 모델
    ↓ eval_cross_encoder.py
results_rerank.json (Bi+Cross 성능)
logs/*_wrongs.txt (오답 분석)
```

---

## 핵심 개념 정리

### 1. 성취기준 (Achievement Standard)
- 교육과정에서 정의한 학습 목표
- 형식: `[코드] 내용` (예: `[6과01-01] 물체의 운동 관찰...`)
- 이 프로젝트의 목표: 교육 텍스트를 올바른 성취기준에 매칭

### 2. Bi-Encoder
- 두 텍스트를 각각 벡터로 변환 후 코사인 유사도 계산
- 장점: 빠름 (사전 계산 가능)
- 단점: 상호작용 없어 정확도 낮음

### 3. Cross-Encoder
- 두 텍스트를 함께 입력하여 직접 관련성 점수 계산
- 장점: 정확함 (Attention으로 상호작용)
- 단점: 느림 (모든 쌍을 매번 계산)

### 4. 2단계 검색
- Bi-Encoder: 전체에서 후보 추출 (Recall 중시)
- Cross-Encoder: 후보를 정밀 재순위화 (Precision 중시)
- 속도와 정확도의 균형

### 5. 주요 평가 메트릭
- **Top-K Accuracy**: 상위 K개 예측 중 정답 포함률
- **MRR (Mean Reciprocal Rank)**: 정답의 평균 역순위
  - Top-1에 정답이 많을수록 높음
  - 범위: 0~1 (1이 완벽)

---

## 모델 정보

### Bi-Encoder
- **jhgan/ko-sroberta-multitask**
- 한국어 문장 임베딩 모델
- RoBERTa 기반

### Cross-Encoder
- **bongsoo/albert-small-kor-cross-encoder-v1**
- 한국어 경량 Cross-Encoder
- ALBERT 기반 (파라미터 적음)

---

## 주요 라이브러리

- **sentence-transformers**: Bi/Cross Encoder 구현
- **pandas**: CSV 데이터 처리
- **torch**: 딥러닝 프레임워크
- **chardet**: 인코딩 자동 감지
- **tqdm**: 진행 상황 표시
- **sklearn**: 평가 메트릭 계산

---

## 파일 구조 요약

```
KorEduBench/
├── dataset/
│   ├── extract_standards.py       # ZIP → 성취기준 CSV
│   ├── add_text_to_standards.py   # 텍스트 샘플 추가
│   └── split_subject.py           # 과목별 분할
├── cosine_similarity/
│   ├── eval_cosine_similarity.py  # 단일 CSV 평가
│   └── batch_cosine_similarity.py # 폴더 일괄 평가
├── cross_encoder/
│   ├── finetune_cross_encoder.py  # Cross-Encoder 학습
│   └── eval_cross_encoder.py      # 재순위화 평가
└── DEVELOPMENT.md                 # 실행 가이드
```

