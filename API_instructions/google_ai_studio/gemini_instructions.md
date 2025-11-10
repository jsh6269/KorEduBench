# Gemini API 사용 방법

## 1. API 키 발급
[Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키 발급

## 2. 패키지 설치
```bash
pip install google-generativeai python-dotenv
```

## 3. `.env` 파일 생성
프로젝트 폴더에 `.env` 파일을 만들고 API 키 입력:
```
GOOGLE_API_KEY=your-api-key-here
```

## 4. 코드 실행
```bash
python gemini_example.py
```
