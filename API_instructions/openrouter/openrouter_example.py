import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에 키 설정
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter 클라이언트 생성
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# 테스트 프롬프트
prompt = """
How many people live in South Korea?
"""

# API 호출
response = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct:free",  # 여기다가 무료 모델 찾아서 넣기
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)
