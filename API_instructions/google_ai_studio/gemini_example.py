import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일에 키 설정
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 설정
genai.configure(api_key=api_key)

# 모델 선택
model = genai.GenerativeModel("gemini-2.5-flash")

# 테스트 프롬프트
prompt = """
How many people live in South Korea?
"""

response = model.generate_content(prompt)
print(response.text.strip())