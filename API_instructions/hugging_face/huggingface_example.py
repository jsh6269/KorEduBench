import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# .env 파일에 키 설정
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# 클라이언트 생성
client = InferenceClient(token=api_key)

# 테스트 프롬프트
prompt = """
How many people live in South Korea?
"""

# API 호출 (다양한 모델 사용 가능)

response = client.text_generation(
    prompt=prompt.strip(),
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=500,
    temperature=0.7
)

print(response)