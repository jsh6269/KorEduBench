import os
from dotenv import load_dotenv
import anthropic

# .env 파일에 키 설정
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# 클라이언트 생성
client = anthropic.Anthropic(api_key=api_key)

# 테스트 프롬프트
prompt = """
How many people live in South Korea?
"""

# API 호출
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(message.content[0].text)
