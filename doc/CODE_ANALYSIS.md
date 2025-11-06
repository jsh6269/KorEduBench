# KorEduBench 코드 상세 분석

이 문서는 KorEduBench 프로젝트의 각 코드 파일을 상세히 분석하여, 각 파일이 수행하는 작업을 단계별로 설명합니다.

---

## 목차

1. [데이터 전처리 파이프라인](#1-데이터-전처리-파이프라인)
   - 1.1 [extract_standards.py](#11-extract_standardspy)
   - 1.2 [add_text_to_standards.py](#12-add_text_to_standardspy)
   - 1.3 [verify_not_empty.py](#13-verify_not_emptypy)
   - 1.4 [check_insufficient_text.py](#14-check_insufficient_textpy)
   - 1.5 [add_additional_text_to_standards.py](#15-add_additional_text_to_standardspy)
   - 1.6 [filter_standards.py](#16-filter_standardspy)
   - 1.7 [split_subject.py](#17-split_subjectpy)

2. [Cosine Similarity 평가](#2-cosine-similarity-평가)
   - 2.1 [eval_cosine_similarity.py](#21-eval_cosine_similaritypy)
   - 2.2 [batch_cosine_similarity.py](#22-batch_cosine_similaritypy)

3. [Cross Encoder 학습 및 평가](#3-cross-encoder-학습-및-평가)
   - 3.1 [finetune_cross_encoder.py](#31-finetune_cross_encoderpy)
   - 3.2 [eval_cross_encoder.py](#32-eval_cross_encoderpy)

4. [LLM 텍스트 분류](#4-llm-텍스트-분류)
   - 4.1 [eval_llm.py](#41-eval_llmpy)
   - 4.2 [finetune_llm.py](#42-finetune_llmpy)
   - 4.3 [eval_finetune_llm.py](#43-eval_finetune_llmpy)

---

## 1. 데이터 전처리 파이프라인

### 1.1 extract_standards.py

**위치**: `src/preprocessing/extract_standards.py`

**목적**: AI Hub 데이터셋의 ZIP 파일들로부터 2022 성취기준(achievement standard)을 추출하여 CSV로 저장

#### 주요 기능 분석

**함수**: `extract_unique_standards(label_dir, output_csv)`

**단계별 처리 과정**:

1. **ZIP 파일 수집**
   - `label_dir` 디렉토리를 순회하며 모든 `.zip` 파일 수집
   - `os.walk()`로 하위 디렉토리까지 재귀적으로 탐색

2. **ZIP 파일 처리**
   - 각 ZIP 파일 내부의 JSON 파일들을 순회
   - JSON에서 다음 정보 추출:
     - `source_data_info.2022_achievement_standard`: 성취기준 리스트
     - `raw_data_info`: subject(과목), school(학교급), grade(학년)
   
3. **성취기준 파싱**
   - 형식: `"[코드] 내용"` (예: `"[6과01-01] 물체의 운동을 관찰하여..."`)
   - `[` 와 `]` 사이의 텍스트를 코드로 추출
   - `]` 이후의 텍스트를 내용(content)으로 추출
   - 중복 제거: `unique_standards` 딕셔너리에 코드를 키로 사용

4. **CSV 저장**
   - 컬럼: `subject`, `school`, `grade`, `code`, `content`
   - 정렬 기준: 과목(subject) → 코드(code) 순
   - UTF-8-sig 인코딩으로 저장 (Excel 호환)

**입력**: 
- `label_dir`: ZIP 파일들이 있는 디렉토리 경로

**출력**: 
- `unique_achievement_standards.csv`: 고유한 성취기준 목록

**예시 실행**:
```bash
python extract_standards.py ./Training/label
```

---

### 1.2 add_text_to_standards.py

**위치**: `src/preprocessing/add_text_to_standards.py`

**목적**: 추출된 성취기준 CSV에 실제 교육 텍스트 샘플들을 매칭하여 추가

#### 주요 기능 분석

**함수**: `append_texts_to_csv(label_dir, csv_path, output_csv, max_texts)`

**단계별 처리 과정**:

1. **CSV 로딩 및 준비**
   - `chardet`로 CSV 인코딩 자동 감지
   - pandas DataFrame으로 CSV 로딩
   - `text_1`, `text_2`, ..., `text_N` 컬럼을 `max_texts` 개수만큼 생성
   - 기존에 `text_` 컬럼이 있다면 이어서 추가

2. **코드-인덱스 매핑 생성**
   - `code_to_idx`: 성취기준 코드를 DataFrame 행 인덱스로 매핑
   - 빠른 검색을 위한 해시맵 구조

3. **ZIP 파일 순회 및 텍스트 추출**
   - 각 JSON 파일에서 학습 데이터 추출:
     - `learning_data_info.text_description`: 텍스트 설명
     - `learning_data_info.text_qa`: 질문-답변 텍스트
     - `learning_data_info.text_an`: 추가 텍스트
   - 세 가지 텍스트를 공백으로 결합하여 `combined_text` 생성

4. **텍스트-성취기준 매칭**
   - JSON의 `2022_achievement_standard`에서 코드 추출
   - 해당 코드가 CSV에 있으면 빈 `text_` 컬럼에 텍스트 추가
   - 이미 `max_texts`만큼 채워진 경우 건너뜀

5. **결과 저장**
   - 업데이트된 DataFrame을 CSV로 저장
   - UTF-8-sig 인코딩 사용

**입력**: 
- `label_dir`: ZIP 파일 디렉토리 (Training/label)
- `csv_path`: 입력 CSV (성취기준만 있는 파일)
- `max_texts`: 추가할 최대 텍스트 개수 (기본값: 160)

**출력**: 
- `text_achievement_standards.csv`: 텍스트 샘플이 추가된 CSV

**예시 실행**:
```bash
python add_text_to_standards.py ./Training/label --csv_path unique_achievement_standards.csv --max_texts 160
```

---

### 1.3 verify_not_empty.py

**위치**: `src/preprocessing/verify_not_empty.py`

**목적**: `text_` 컬럼들이 순서대로 채워져 있는지 검증 (중간에 빈 컬럼이 없는지 확인)

#### 주요 기능 분석

**함수**: `verify_text_order(input_csv)`

**단계별 처리 과정**:

1. **CSV 로딩**
   - 인코딩 자동 감지 및 DataFrame 로딩
   - 모든 `text_` 컬럼을 찾아 숫자 순으로 정렬

2. **순서 검증**
   - 각 행에 대해 첫 번째 빈 `text_` 컬럼을 찾음
   - 그 이후의 컬럼에 데이터가 있는지 확인
   - 이슈 발견 시 기록 (예: `text_5`가 비었는데 `text_10`에 데이터가 있음)

3. **결과 출력**
   - 검증 통과/실패 여부
   - 과목별 이슈 요약
   - 처음 10개 이슈 상세 출력

**입력**: 
- `input_csv`: 검증할 CSV 파일 경로

**출력**: 
- 콘솔에 검증 결과 출력
- exit code (0: 성공, 1: 실패)

**예시 실행**:
```bash
python verify_not_empty.py text_achievement_standards.csv
```

---

### 1.4 check_insufficient_text.py

**위치**: `src/preprocessing/check_insufficient_text.py`

**목적**: 텍스트 샘플이 충분하지 않은 성취기준들을 찾아 별도 CSV로 저장

#### 주요 기능 분석

**함수**: `check_insufficient_text(input_csv, output_csv, min_texts)`

**단계별 처리 과정**:

1. **CSV 로딩 및 분석**
   - 인코딩 자동 감지 및 로딩
   - 모든 `text_` 컬럼 찾기

2. **텍스트 개수 카운팅**
   - 각 행의 비어있지 않은 `text_` 컬럼 개수 계산
   - NaN, 빈 문자열, 공백 문자열은 비어있음으로 간주

3. **통계 출력**
   - 텍스트 개수별 행 분포 (0개, 1-20개, 21-40개, ...)
   - 과목별/학교급별 충분/불충분 통계
   - 평균, 중앙값, 최소/최대 텍스트 개수

4. **부족한 행 필터링**
   - `min_texts` 미만인 행들만 선택
   - `subject`, `school`, `grade`, `code`, `content`, `text_count` 컬럼 저장

5. **결과 저장**
   - `insufficient_text.csv`로 저장
   - 과목별, 학교급별 통계 출력

**입력**: 
- `input_csv`: 입력 CSV 파일 (기본값: `text_achievement_standards.csv`)
- `min_texts`: 최소 텍스트 개수 (기본값: 160)

**출력**: 
- `insufficient_text.csv`: 텍스트가 부족한 성취기준 목록

**예시 실행**:
```bash
python check_insufficient_text.py --min_texts 160
```

---

### 1.5 add_additional_text_to_standards.py

**위치**: `src/preprocessing/add_additional_text_to_standards.py`

**목적**: `insufficient_text.csv`에 있는 성취기준들에 대해 Validation 데이터셋에서 추가 텍스트 샘플 추가

#### 주요 기능 분석

**함수**: `add_additional_texts_to_csv(...)`

**단계별 처리 과정**:

1. **부족 성취기준 로딩**
   - `insufficient_text.csv` 로딩
   - (code, content) 튜플의 집합(set) 생성하여 빠른 검색 지원

2. **기존 CSV 로딩**
   - `text_achievement_standards.csv` 로딩
   - 필요 시 추가 `text_` 컬럼 생성

3. **Validation ZIP 파일 순회**
   - Validation/label 디렉토리의 ZIP 파일들 처리
   - JSON에서 `text_description` 추출
   - 성취기준 코드가 부족 목록에 있는지 확인

4. **텍스트 추가**
   - 부족 목록에 있는 성취기준에만 텍스트 추가
   - 첫 번째 빈 `text_` 컬럼에 추가
   - `max_texts`에 도달하면 해당 성취기준을 부족 목록에서 제거

5. **결과 저장**
   - 업데이트된 DataFrame을 원래 경로에 덮어쓰기
   - 추가된 텍스트 개수 출력

**입력**: 
- `label_dir`: Validation ZIP 파일 디렉토리
- `insufficient_csv`: 부족 성취기준 목록
- `text_standards_csv`: 업데이트할 CSV
- `max_texts`: 최대 텍스트 컬럼 수

**출력**: 
- `text_achievement_standards.csv` (업데이트됨)

**예시 실행**:
```bash
python add_additional_text_to_standards.py ./Validation/label --max_texts 160
```

---

### 1.6 filter_standards.py

**위치**: `src/preprocessing/filter_standards.py`

**목적**: 충분한 텍스트가 있는 성취기준만 선택하고, train/valid로 분할

#### 주요 기능 분석

**함수**: `split_texts_to_train_valid(...)`

**단계별 처리 과정**:

1. **CSV 로딩**
   - `text_achievement_standards.csv` 로딩
   - `text_1` ~ `text_{num_texts}` 범위 확인

2. **필터링**
   - 각 행의 `text_1` ~ `text_{num_texts}` 범위에서 비어있지 않은 컬럼 개수 확인
   - `num_texts` 개 이상의 텍스트가 있는 행만 선택

3. **텍스트 전처리**
   - HTML 테이블 처리: `<td>` 태그 내용만 추출
   - 줄바꿈을 공백으로 변환
   - 다중 공백을 단일 공백으로 변환

4. **Train/Valid 분할**
   - 각 행에서 `num_texts`개의 텍스트를 랜덤 샘플링
   - 앞의 `num_texts/2`개는 train으로
   - 뒤의 `num_texts/2`개는 valid로
   - **중요**: 동일 성취기준의 train과 valid 텍스트는 완전히 분리됨

5. **결과 저장**
   - `train.csv`: 각 성취기준당 `num_texts/2`개의 텍스트
   - `valid.csv`: 각 성취기준당 `num_texts/2`개의 텍스트
   - 메타 컬럼 + `text_1` ~ `text_{num_texts/2}` 구조

**입력**: 
- `input_csv`: 입력 CSV (기본값: `text_achievement_standards.csv`)
- `num_texts`: 샘플링할 텍스트 개수 (짝수여야 함, 기본값: 160)
- `train_csv`: train 출력 경로
- `valid_csv`: valid 출력 경로
- `seed`: 랜덤 시드 (기본값: 42)

**출력**: 
- `train.csv`: 학습용 데이터
- `valid.csv`: 검증용 데이터

**예시 실행**:
```bash
python filter_standards.py --num_texts 160 --input_csv text_achievement_standards.csv --train_csv train.csv --valid_csv valid.csv
```

---

### 1.7 split_subject.py

**위치**: `src/preprocessing/split_subject.py`

**목적**: train.csv와 valid.csv를 각각 과목별로 분할하여 개별 파일로 저장

#### 주요 기능 분석

**함수**: `split_csv_by_subject(input_path, output_folder, max_texts, encoding)`

**단계별 처리 과정**:

1. **출력 디렉토리 생성**
   - 형식: `{output_folder}` (예: `train_80`, `valid_80`)
   - 이미 존재해도 오류 없음 (`exist_ok=True`)

2. **CSV 로딩 및 컬럼 선택**
   - 기본 컬럼: `subject`, `school`, `grade`, `code`, `content`
   - 텍스트 컬럼: `text_1` ~ `text_{max_texts}`
   - 존재하는 컬럼만 선택 (일부만 있어도 동작)

3. **과목별 그룹화 및 저장**
   - `df.groupby("subject")`로 과목별로 데이터 분할
   - 파일명 안전화: 영숫자와 `_`만 허용, 공백은 `_`로 변환
   - 각 과목을 `{과목명}.csv`로 저장
   - 예시: `과학.csv`, `수학.csv`, `영어.csv`

**입력**: 
- `input_path`: 통합 CSV 파일 경로 (`train.csv` 또는 `valid.csv`)
- `output_folder`: 출력 폴더명 (예: `train_80`)
- `max_texts`: 포함할 텍스트 컬럼 개수
- `encoding`: CSV 인코딩 (기본값: utf-8-sig)

**출력**: 
- `{output_folder}/` 디렉토리 내 과목별 CSV 파일들

**예시 실행**:
```bash
python split_subject.py --input train.csv --output train_80 --max-texts 80
python split_subject.py --input valid.csv --output valid_80 --max-texts 80
```

---

## 2. Cosine Similarity 평가

### 2.1 eval_cosine_similarity.py

**위치**: `src/cosine_similarity/eval_cosine_similarity.py`

**목적**: Bi-Encoder를 사용하여 코사인 유사도 기반 검색 성능 평가 (베이스라인)

#### 주요 기능 분석

**함수**: `evaluate_cosine_similarity_baseline(...)`

**단계별 처리 과정**:

1. **데이터 준비**
   - CSV 인코딩 자동 감지 (`detect_encoding`)
   - 필수 컬럼 확인: `code`, `content`
   - `text_` 컬럼들 찾기 (평가용 샘플)
   - `max_samples_per_row` 자동 계산 (각 행의 최대 텍스트 개수)

2. **Bi-Encoder 모델 로딩**
   - `SentenceTransformer` 모델 로딩
   - 기본 모델: `jhgan/ko-sroberta-multitask` (한국어 문장 임베딩)
   - 평가 모드로 설정

3. **성취기준 인코딩**
   - 모든 성취기준의 `content` 필드를 벡터로 변환
   - GPU 텐서로 변환 (`convert_to_tensor=True`)
   - 진행 상황 표시 (`show_progress_bar=True`)

4. **샘플 텍스트 수집 및 인코딩**
   - 각 행의 `text_1`, `text_2`, ... 컬럼에서 텍스트 추출
   - `max_samples_per_row` 제한 적용
   - 샘플과 정답 코드 쌍 생성
   - 모든 샘플 텍스트를 벡터로 인코딩

5. **코사인 유사도 계산**
   - `util.cos_sim(emb_samples, emb_contents)`: 유사도 행렬 계산
   - 크기: `[num_samples, num_standards]`
   - 각 샘플이 모든 성취기준과 얼마나 유사한지 계산

6. **Top-K 정확도 계산**
   - Top-1, 3, 10, 20, 40, 60 정확도 계산
   - 각 샘플에 대해 상위 k개 예측 중 정답이 있는지 확인
   - 정확도 = 맞춘 샘플 수 / 전체 샘플 수

7. **MRR (Mean Reciprocal Rank) 계산**
   - 각 샘플의 정답이 몇 번째 순위인지 확인
   - Reciprocal Rank = 1 / 순위
   - MRR = 모든 샘플의 RR 평균
   - 높을수록 정답이 상위에 랭크됨을 의미

8. **결과 저장**
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
python eval_cosine_similarity.py --input_csv ../dataset/valid_80/과학.csv
```

---

### 2.2 batch_cosine_similarity.py

**위치**: `src/cosine_similarity/batch_cosine_similarity.py`

**목적**: 폴더 내 모든 CSV 파일에 대해 코사인 유사도 평가를 일괄 실행

#### 주요 기능 분석

**함수**: `evaluate_folder(...)`

**단계별 처리 과정**:

1. **CSV 파일 수집**
   - `folder_path` 내 모든 `.csv` 파일 찾기
   - 파일이 없으면 경고 메시지 출력

2. **각 CSV 파일 평가**
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
python batch_cosine_similarity.py --folder_path ../dataset/valid_80/
```

---

## 3. Cross Encoder 학습 및 평가

### 3.1 finetune_cross_encoder.py

**위치**: `src/cross_encoder/finetune_cross_encoder.py`

**목적**: Cross-Encoder 모델을 한국어 교육 데이터에 파인튜닝

#### 주요 기능 분석

**함수**: `fine_tune_cross_encoder(...)`

#### Cross-Encoder란?
- Bi-Encoder: 텍스트를 별도로 인코딩 후 유사도 계산 (빠르지만 정확도 낮음)
- Cross-Encoder: 두 텍스트를 함께 입력하여 직접 관련성 점수 계산 (느리지만 정확)

**단계별 처리 과정**:

1. **데이터 로딩 및 분할**
   - CSV에서 성취기준 데이터 로딩
   - Train/Test 분할 (기본 8:2)
   - 행(row) 단위로 분할하여 데이터 누수 방지

2. **학습 데이터 쌍 생성** (`build_pairs_from_df`)
   
   **긍정 쌍(Positive Pairs) 생성**:
   - 각 성취기준(`content`)과 해당 텍스트 샘플들 매칭
   - 형식: `(text_sample, achievement_standard_content)` → label=1.0
   - 중복 제거를 위해 `pos_keys` 집합 사용
   
   **부정 쌍(Negative Pairs) 생성**:
   - 랜덤하게 텍스트와 성취기준을 매칭 (관련 없는 쌍)
   - 형식: `(text_sample, random_content)` → label=0.0
   - 긍정 쌍과 겹치지 않도록 검증
   - 부정 쌍 개수 = 긍정 쌍 개수 × `neg_ratio`
   - 최대 시도 횟수: `num_neg * 20`

3. **모델 초기화**
   - 기본 모델: `bongsoo/albert-small-kor-cross-encoder-v1`
   - 한국어에 특화된 경량 Cross-Encoder
   - `num_labels=1`: 연속적인 관련성 점수 출력 (0~1)

4. **학습 설정**
   - DataLoader 생성 (배치 처리)
   - Warmup steps: 전체 스텝의 10%
   - Learning rate 서서히 증가 후 감소 (학습 안정화)

5. **파인튜닝 실행**
   - `model.fit()` 호출
   - 에폭 수, Learning rate, Warmup 설정
   - 진행 상황 표시
   - 학습된 모델을 `output_dir`에 저장

6. **평가**
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
python finetune_cross_encoder.py --input_csv ../dataset/train_80/과학.csv --epochs 3
```

---

### 3.2 eval_cross_encoder.py

**위치**: `src/cross_encoder/eval_cross_encoder.py`

**목적**: Bi-Encoder로 후보를 추출한 후 Cross-Encoder로 재순위화하여 정확도 향상

#### 주요 기능 분석

**함수**: `evaluate_bi_cross_pipeline(...)`

#### 2단계 검색 파이프라인
1. **Bi-Encoder**: 빠르게 상위 K개 후보 추출
2. **Cross-Encoder**: 후보들을 정밀하게 재순위화

**단계별 처리 과정**:

1. **데이터 및 모델 준비**
   - CSV 로딩 및 검증
   - Bi-Encoder 로딩 (`jhgan/ko-sroberta-multitask`)
   - Cross-Encoder 로딩 (파인튜닝된 모델)
   - 성취기준 벡터화 (Bi-Encoder)

2. **샘플 수집**
   - `text_` 컬럼에서 평가 샘플 추출
   - 정답 코드와 함께 저장
   - `max_samples_per_row` 제한 적용

3. **1단계: Bi-Encoder 검색**
   - 모든 샘플을 벡터로 인코딩
   - 코사인 유사도 계산
   - 각 샘플당 상위 `top_k`개 후보 선택 (기본값: 20)
   - 빠른 속도로 후보군 축소

4. **2단계: Cross-Encoder 재순위화**
   
   각 샘플에 대해:
   - `top_k`개 후보와 query를 쌍으로 만듦
   - Cross-Encoder로 정확한 관련성 점수 계산
   - 점수 기준으로 재정렬
   - Top-1 예측이 틀린 경우 오답 샘플로 기록

5. **평가 메트릭 계산**
   - **Top-1/3/10/20 Accuracy**: 상위 K개 내 정답 포함 비율
   - **MRR**: 평균 역순위
   - Bi-Encoder 단독보다 일반적으로 높은 성능

6. **결과 저장**
   
   **JSON 로깅**:
   - 모델 정보, 과목, 정확도, MRR
   - 기존 결과 업데이트 또는 추가
   - `results_rerank.json`에 저장
   
   **오답 분석**:
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
python eval_cross_encoder.py --input_csv ../dataset/valid_80/과학.csv --cross_model ./cross_finetuned
```

---

## 4. LLM 텍스트 분류

### 4.1 eval_llm.py

**위치**: `src/llm_text_classification/eval_llm.py`

**목적**: 사전학습된 생성형 LLM을 사용하여 교육 텍스트를 성취기준으로 분류

#### 개요

기존의 Bi-Encoder나 Cross-Encoder와 달리, 생성형 LLM을 사용하여 직접 성취기준 코드를 생성합니다.

**접근 방법**:
- 프롬프트에 모든 후보 성취기준을 나열
- LLM이 가장 관련성 높은 성취기준 코드를 직접 출력
- 예: `10영03-04`

#### 주요 기능 분석

**함수**: `evaluate_llm_classification(...)`

**단계별 처리 과정**:

1. **데이터 로딩**
   - CSV에서 성취기준 및 샘플 텍스트 로딩
   - `max_samples_per_row`: 각 성취기준당 평가할 샘플 수
   - `max_total_samples`: 전체 샘플 수 제한 (랜덤 샘플링)

2. **후보 준비**
   - 모든 성취기준을 후보 리스트로 준비
   - `max_candidates`: 후보 수 제한 (기본값: 200)
   - 성취기준이 많은 경우 랜덤 샘플링
   - 형식: `[(번호, 코드, 내용), ...]`

3. **LLM 모델 로딩**
   - 기본 모델: `Qwen/Qwen2.5-3B-Instruct`
   - float16 정밀도로 로딩 (GPU)
   - 평가 모드로 설정

4. **프롬프트 생성**
   - 각 샘플에 대해 분류 프롬프트 생성
   - 프롬프트 구조:
     ```
     다음 교육 내용을 가장 잘 설명하는 성취기준 코드를 선택하세요.
     
     교육 내용:
     [샘플 텍스트]
     
     후보 성취기준:
     1. [6과01-01] 물체의 운동을 관찰하여...
     2. [6과01-02] 자석의 성질을 탐구하여...
     ...
     
     정답 코드:
     ```

5. **예측 생성**
   - 각 샘플에 대해 LLM이 코드 생성
   - `max_new_tokens`: 생성할 최대 토큰 수 (기본값: 50)
   - `temperature`: 샘플링 온도 (기본값: 0.1, 거의 탐욕적)
   - `max_input_length`: 입력 최대 길이 (초과 시 자동 truncate)

6. **응답 파싱**
   - LLM 출력에서 성취기준 코드 추출
   - 매칭 타입 분류:
     - **exact**: 정확히 일치하는 코드 (예: `10영03-04`)
     - **partial**: 부분 일치 (예: `03-04`를 `10영03-04`로 복원)
     - **invalid**: 유효하지 않은 코드

7. **평가**
   - **Accuracy**: 정답 비율
   - **MRR**: 평균 역순위 (LLM은 단일 예측만 하므로 Accuracy와 동일)
   - **Exact Match %**: 정확히 일치하는 응답 비율
   - **Match Type 분포**: exact/partial/invalid 비율

8. **결과 저장**
   - JSON 로깅: `results.json`
   - 오답 샘플 저장: `logs/{과목명}_wrongs.txt` (최대 100개)
   - 정답 샘플 저장: `logs/{과목명}_corrects.txt` (최대 100개)

**입력**: 
- `input_csv`: 평가할 CSV 파일
- `model_name`: LLM 모델 이름 (기본값: `Qwen/Qwen2.5-3B-Instruct`)
- `max_samples_per_row`: 행당 최대 샘플 수
- `max_total_samples`: 전체 최대 샘플 수
- `max_candidates`: 최대 후보 개수 (기본값: 200)
- `max_new_tokens`: 생성 최대 토큰 수 (기본값: 50)
- `temperature`: 샘플링 온도 (기본값: 0.1)
- `max_input_length`: 입력 최대 길이 (기본값: 6144)

**출력**: 
- `output/llm_text_classification/results.json`: 평가 결과
- `output/llm_text_classification/logs/{과목명}_wrongs.txt`: 오답 샘플
- `output/llm_text_classification/logs/{과목명}_corrects.txt`: 정답 샘플

**예시 실행**:
```bash
python eval_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --max-candidates 120 \
    --max-total-samples 100
```

---

### 4.2 finetune_llm.py

**위치**: `src/llm_text_classification/finetune_llm.py`

**목적**: 생성형 LLM을 교육 성취기준 분류 태스크에 파인튜닝

#### 개요

Unsloth 라이브러리를 사용하여 효율적으로 LLM을 파인튜닝합니다. LoRA (Low-Rank Adaptation) 기법을 사용하여 메모리 효율적으로 학습합니다.

#### 주요 기능 분석

**함수**: `finetune_llm(...)`

**단계별 처리 과정**:

1. **학습 데이터 준비**
   - `train_dir` 내 모든 CSV 파일 로딩 (과목별 파일들)
   - 각 파일에서 성취기준과 텍스트 샘플 추출
   - `max_samples_per_row`: 각 성취기준당 사용할 샘플 수 (기본값: 1)
   - `max_total_samples`: 전체 학습 샘플 수 제한
   - `max_candidates`: 프롬프트당 후보 개수 제한 (기본값: 120)

2. **프롬프트 생성**
   - 각 샘플에 대해 평가와 동일한 형식의 프롬프트 생성
   - 정답 코드를 completion으로 추가
   - 형식:
     ```
     [프롬프트]
     정답 코드:
     [정답 코드]
     ```

3. **모델 로딩 (Unsloth)**
   - 기본 모델: `unsloth/Qwen2.5-3B-Instruct`
   - 4-bit 양자화 로딩 (메모리 절약)
   - `max_seq_length`: 최대 시퀀스 길이 (기본값: 6144)

4. **LoRA 어댑터 추가**
   - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
   - `lora_r`: LoRA rank (기본값: 16)
   - `lora_alpha`: LoRA alpha (기본값: 16)
   - `lora_dropout`: LoRA dropout (기본값: 0.0)

5. **학습 설정**
   - `SFTTrainer` (Supervised Fine-Tuning Trainer) 사용
   - 하이퍼파라미터:
     - `num_train_epochs`: 에폭 수 (기본값: 1)
     - `per_device_train_batch_size`: 배치 크기 (기본값: 4)
     - `gradient_accumulation_steps`: 그래디언트 누적 (기본값: 4)
     - `learning_rate`: 학습률 (기본값: 2e-4)
     - `warmup_steps`: 워밍업 스텝 (기본값: 5)
   - Optimizer: AdamW 8-bit
   - Mixed Precision: FP16 또는 BF16 (GPU 지원 여부에 따라)

6. **학습 실행**
   - `trainer.train()` 호출
   - 진행 상황 자동 출력
   - 체크포인트 자동 저장

7. **모델 저장**
   - **LoRA 어댑터**: `output_dir/` (경량, 재사용 가능)
   - **Merged 16-bit 모델**: `output_dir/merged_16bit/` (추론용, 고품질)
   - **Merged 4-bit 모델**: `output_dir/merged_4bit/` (추론용, 메모리 절약)
   - **학습 정보**: `output_dir/training_info.json`

**입력**: 
- `train_dir`: 학습용 CSV 파일들이 있는 디렉토리
- `model_name`: 기본 LLM 모델 (기본값: `unsloth/Qwen2.5-3B-Instruct`)
- `output_dir`: 출력 디렉토리 (기본값: `model/finetuned_llm`)
- `max_seq_length`: 최대 시퀀스 길이 (기본값: 6144)
- `max_samples_per_row`: 행당 최대 샘플 수 (기본값: 1)
- `max_total_samples`: 전체 최대 샘플 수 (기본값: None)
- `max_candidates`: 후보 최대 개수 (기본값: 120)
- 학습 하이퍼파라미터들

**출력**: 
- `model/finetuned_llm/`: LoRA 어댑터
- `model/finetuned_llm/merged_16bit/`: 병합된 16-bit 모델
- `model/finetuned_llm/merged_4bit/`: 병합된 4-bit 모델
- `model/finetuned_llm/training_info.json`: 학습 정보

**예시 실행**:
```bash
python finetune_llm.py \
    --train_dir ../dataset/train_80 \
    --model_name unsloth/Qwen2.5-3B-Instruct \
    --output_dir ../model/finetuned_llm \
    --num-train-epochs 1 \
    --max-samples-per-row 1 \
    --max-candidates 120
```

---

### 4.3 eval_finetune_llm.py

**위치**: `src/llm_text_classification/eval_finetune_llm.py`

**목적**: 파인튜닝된 LLM을 사용하여 교육 텍스트 분류 평가

#### 개요

`finetune_llm.py`로 학습된 모델을 로딩하여 평가합니다. `eval_llm.py`와 거의 동일한 평가 프로세스를 사용하지만, 파인튜닝된 모델을 로딩하는 방식이 다릅니다.

#### 주요 기능 분석

**함수**: `evaluate_finetuned_llm(...)`

**단계별 처리 과정**:

1. **데이터 로딩**
   - `eval_llm.py`와 동일

2. **파인튜닝된 모델 로딩**
   - Unsloth의 `FastLanguageModel.from_pretrained()` 사용
   - 로딩 옵션:
     - `use_merged=True` (기본값): `merged_16bit/` 모델 로딩
     - `use_merged=False`: LoRA 어댑터 로딩
   - `FastLanguageModel.for_inference()`: 추론 최적화 (2배 빠름)

3. **학습 정보 로딩**
   - `training_info.json`에서 학습 설정 읽기
   - 기본 모델, 학습 샘플 수, 에폭 수 등 출력

4. **평가**
   - `eval_llm.py`와 동일한 프로세스
   - 프롬프트 생성 → LLM 예측 → 응답 파싱 → 평가

5. **결과 저장**
   - JSON 로깅: `finetuned_results.json`
   - 오답 샘플: `finetuned_logs/{모델명}_{과목명}_wrongs.txt`
   - 정답 샘플: `finetuned_logs/{모델명}_{과목명}_corrects.txt`

**입력**: 
- `input_csv`: 평가할 CSV 파일
- `model_path`: 파인튜닝된 모델 디렉토리
- `use_merged`: merged 모델 사용 여부 (기본값: True)
- 나머지는 `eval_llm.py`와 동일

**출력**: 
- `output/llm_text_classification/finetuned_results.json`: 평가 결과
- `output/llm_text_classification/finetuned_logs/`: 오답/정답 샘플

**예시 실행**:
```bash
python eval_finetune_llm.py \
    --input_csv ../dataset/valid_80/과학.csv \
    --model_path ../model/finetuned_llm \
    --max-candidates 120 \
    --max-total-samples 100
```

---

## 전체 파이프라인 요약

### Phase 1: 데이터 준비
```
원본 ZIP 파일 (Training/label)
    ↓ extract_standards.py
unique_achievement_standards.csv (성취기준만)
    ↓ add_text_to_standards.py (Training)
text_achievement_standards.csv (텍스트 샘플 추가, 일부 부족)
    ↓ verify_not_empty.py (선택적)
검증 완료
    ↓ check_insufficient_text.py
insufficient_text.csv (텍스트 부족 성취기준 목록)
    ↓ add_additional_text_to_standards.py (Validation)
text_achievement_standards.csv (업데이트, 텍스트 추가)
    ↓ check_insufficient_text.py (재확인)
충분도 확인
    ↓ filter_standards.py
train.csv + valid.csv (train/valid 분할)
    ↓ split_subject.py
train_80/과학.csv, 수학.csv, ... (과목별 분할)
valid_80/과학.csv, 수학.csv, ... (과목별 분할)
```

### Phase 2: 베이스라인 평가 (Bi-Encoder)
```
valid_80/*.csv
    ↓ eval_cosine_similarity.py (또는 batch)
output/cosine_similarity/results.json
```

### Phase 3: Cross-Encoder 파인튜닝 및 평가
```
train_80/*.csv
    ↓ finetune_cross_encoder.py
cross_finetuned/ (파인튜닝된 모델)
    ↓ eval_cross_encoder.py
output/cross_encoder/results_rerank.json
output/cross_encoder/logs/*_wrongs.txt
```

### Phase 4: LLM 베이스라인 평가
```
valid_80/*.csv
    ↓ eval_llm.py
output/llm_text_classification/results.json
output/llm_text_classification/logs/*_wrongs.txt
output/llm_text_classification/logs/*_corrects.txt
```

### Phase 5: LLM 파인튜닝 및 평가
```
train_80/*.csv
    ↓ finetune_llm.py
model/finetuned_llm/ (LoRA 어댑터)
model/finetuned_llm/merged_16bit/ (병합된 모델)
model/finetuned_llm/merged_4bit/ (4-bit 양자화 모델)
    ↓ eval_finetune_llm.py
output/llm_text_classification/finetuned_results.json
output/llm_text_classification/finetuned_logs/*_wrongs.txt
output/llm_text_classification/finetuned_logs/*_corrects.txt
```

---

## 핵심 개념 정리

### 1. 성취기준 (Achievement Standard)
- 교육과정에서 정의한 학습 목표
- 형식: `[코드] 내용` (예: `[6과01-01] 물체의 운동 관찰...`)
- 이 프로젝트의 목표: 교육 텍스트를 올바른 성취기준에 매칭

### 2. 접근 방법 비교

#### Bi-Encoder (코사인 유사도)
- 두 텍스트를 각각 벡터로 변환 후 코사인 유사도 계산
- 장점: 빠름 (사전 계산 가능), 확장성 좋음
- 단점: 상호작용 없어 정확도 낮음
- 사용 모델: `jhgan/ko-sroberta-multitask`

#### Cross-Encoder (재순위화)
- 두 텍스트를 함께 입력하여 직접 관련성 점수 계산
- 장점: 정확함 (Attention으로 상호작용)
- 단점: 느림 (모든 쌍을 매번 계산)
- 사용 방법: Bi-Encoder로 후보 추출 → Cross-Encoder로 재순위화
- 사용 모델: `bongsoo/albert-small-kor-cross-encoder-v1`

#### LLM (생성형)
- 생성형 LLM이 직접 성취기준 코드를 생성
- 장점: 유연함, 사람처럼 추론 가능, 파인튜닝 효과 큼
- 단점: 느림, 리소스 많이 사용, 프롬프트 길이 제한
- 사용 모델: `Qwen/Qwen2.5-3B-Instruct` (베이스라인), `unsloth/Qwen2.5-3B-Instruct` (파인튜닝)

### 3. 2단계 검색 (Bi-Encoder + Cross-Encoder)
- Bi-Encoder: 전체에서 후보 추출 (Recall 중시)
- Cross-Encoder: 후보를 정밀 재순위화 (Precision 중시)
- 속도와 정확도의 균형

### 4. LoRA (Low-Rank Adaptation)
- 대규모 언어 모델을 효율적으로 파인튜닝하는 기법
- 전체 파라미터를 업데이트하지 않고 작은 어댑터만 학습
- 메모리 사용량 대폭 감소
- 학습 속도 향상

### 5. 주요 평가 메트릭

#### Top-K Accuracy
- 상위 K개 예측 중 정답 포함률
- K=1: 가장 높은 예측만 평가
- K가 클수록 완화된 평가

#### MRR (Mean Reciprocal Rank)
- 정답의 평균 역순위
- Top-1에 정답이 많을수록 높음
- 범위: 0~1 (1이 완벽)
- 계산: RR = 1 / rank, MRR = average(RR)

#### Exact Match Percentage
- LLM이 정확히 유효한 성취기준 코드를 생성한 비율
- Partial match (부분 일치)와 구분

#### Match Type Distribution
- exact: 정확히 일치하는 코드
- partial: 부분 일치 (예: `03-04` → `10영03-04`)
- invalid: 유효하지 않은 코드

---

## 모델 정보

### Bi-Encoder
- **jhgan/ko-sroberta-multitask**
- 한국어 문장 임베딩 모델
- RoBERTa 기반
- 768 차원 벡터 출력

### Cross-Encoder
- **bongsoo/albert-small-kor-cross-encoder-v1**
- 한국어 경량 Cross-Encoder
- ALBERT 기반 (파라미터 적음)
- 단일 관련성 점수 출력 (0~1)

### LLM
- **Qwen/Qwen2.5-3B-Instruct** (베이스라인)
- **unsloth/Qwen2.5-3B-Instruct** (파인튜닝용)
- 3B 파라미터 생성형 언어 모델
- Instruction-tuned (지시사항 따르기 특화)
- 한국어 지원

---

## 주요 라이브러리

### 공통
- **pandas**: CSV 데이터 처리
- **chardet**: 인코딩 자동 감지
- **tqdm**: 진행 상황 표시

### Bi-Encoder / Cross-Encoder
- **sentence-transformers**: Bi/Cross Encoder 구현
- **torch**: 딥러닝 프레임워크
- **sklearn**: 평가 메트릭 계산

### LLM
- **transformers**: Hugging Face Transformers 라이브러리
- **unsloth**: 효율적인 LLM 파인튜닝 라이브러리
- **trl**: Transformer Reinforcement Learning (SFTTrainer)
- **torch**: PyTorch 프레임워크
- **peft**: Parameter-Efficient Fine-Tuning (LoRA 지원)

---

## 파일 구조 요약

```
KorEduBench/
├── src/
│   ├── preprocessing/
│   │   ├── extract_standards.py              # ZIP → 성취기준 CSV
│   │   ├── add_text_to_standards.py          # Training 텍스트 추가
│   │   ├── verify_not_empty.py               # 텍스트 순서 검증
│   │   ├── check_insufficient_text.py        # 부족 텍스트 체크
│   │   ├── add_additional_text_to_standards.py  # Validation 텍스트 추가
│   │   ├── filter_standards.py               # Train/Valid 분할
│   │   └── split_subject.py                  # 과목별 분할
│   ├── cosine_similarity/
│   │   ├── eval_cosine_similarity.py         # 단일 CSV 평가
│   │   └── batch_cosine_similarity.py        # 폴더 일괄 평가
│   ├── cross_encoder/
│   │   ├── finetune_cross_encoder.py         # Cross-Encoder 학습
│   │   └── eval_cross_encoder.py             # 재순위화 평가
│   └── llm_text_classification/
│       ├── eval_llm.py                       # LLM 베이스라인 평가
│       ├── finetune_llm.py                   # LLM 파인튜닝
│       └── eval_finetune_llm.py              # 파인튜닝 LLM 평가
├── scripts/
│   ├── preprocess.sh                         # 전처리 파이프라인 실행
│   ├── cosine_similarity.sh                  # Bi-Encoder 평가 실행
│   ├── cross_encoder.sh                      # Cross-Encoder 실행
│   ├── llm_text_classification.sh            # LLM 베이스라인 실행
│   ├── finetuning_llm.sh                     # LLM 파인튜닝 실행
│   └── finetune_llm_text_classification.sh   # 파인튜닝 LLM 평가 실행
├── dataset/
│   ├── unique_achievement_standards.csv      # 성취기준 목록
│   ├── text_achievement_standards.csv        # 텍스트 추가된 데이터
│   ├── insufficient_text.csv                 # 부족 성취기준 목록
│   ├── train.csv                             # 통합 학습 데이터
│   ├── valid.csv                             # 통합 검증 데이터
│   ├── train_80/                             # 과목별 학습 데이터
│   └── valid_80/                             # 과목별 검증 데이터
├── model/
│   └── finetuned_llm/                        # 파인튜닝된 LLM
│       ├── merged_16bit/                     # 16-bit 병합 모델
│       ├── merged_4bit/                      # 4-bit 병합 모델
│       └── training_info.json                # 학습 정보
└── output/
    ├── cosine_similarity/
    │   └── results.json
    ├── cross_encoder/
    │   ├── results_rerank.json
    │   └── logs/
    └── llm_text_classification/
        ├── results.json                      # LLM 베이스라인 결과
        ├── finetuned_results.json            # 파인튜닝 LLM 결과
        ├── logs/                             # 베이스라인 로그
        └── finetuned_logs/                   # 파인튜닝 로그
```

---

## 실행 가이드 요약

### 1. 데이터 전처리
```bash
cd scripts
bash preprocess.sh
```

### 2. Bi-Encoder 평가
```bash
cd scripts
bash cosine_similarity.sh
```

### 3. Cross-Encoder 학습 및 평가
```bash
cd scripts
bash cross_encoder.sh
```

### 4. LLM 베이스라인 평가
```bash
cd scripts
bash llm_text_classification.sh
```

### 5. LLM 파인튜닝
```bash
cd scripts
bash finetuning_llm.sh
```

### 6. 파인튜닝된 LLM 평가
```bash
cd scripts
bash finetune_llm_text_classification.sh
```

---

