# Hugging Face API 사용 방법

## 1. API 키 발급
[Hugging Face Settings](https://huggingface.co/settings/tokens)에서 API 토큰 발급
- "New token" 클릭
- "Read" 권한 선택

## 2. 패키지 설치
```bash
pip install huggingface-hub python-dotenv
```

## 3. `.env` 파일 생성
프로젝트 폴더에 `.env` 파일을 만들고 API 키 입력:
```
HUGGINGFACE_API_KEY=your-api-key-here
```

## 4. 코드 실행
```bash
python huggingface_example.py
```