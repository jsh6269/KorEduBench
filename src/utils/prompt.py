"""
Prompt templates for LLM-based text classification.
Provides functions to generate prompts for educational content classification.
"""


def create_classification_prompt(text: str, candidates: list[tuple[int, str, str]]) -> str:
    """
    Create a zero-shot classification prompt for educational content matching.
    
    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (index, code, content) representing achievement standards
    
    Returns:
        Formatted prompt string for LLM
    """
    prompt = """당신은 교육과정 전문가입니다. 주어진 교과서 텍스트를 읽고, 가장 적합한 교육 성취기준을 선택해주세요.

# 교과서 텍스트
{text}

# 교육 성취기준 목록
{candidates}

# 지시사항
위 교과서 텍스트의 내용을 가장 잘 설명하는 성취기준의 번호를 하나만 선택하세요.
반드시 숫자만 출력하세요. (예: 1 또는 5 또는 23)

# 답변
"""
    
    # Format candidates
    candidate_text = "\n".join([
        f"{idx}. [{code}] {content}"
        for idx, code, content in candidates
    ])
    
    return prompt.format(text=text, candidates=candidate_text)


def create_simple_classification_prompt(text: str, candidates: list[tuple[int, str, str]]) -> str:
    """
    Create a simpler classification prompt (alternative version).
    
    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (index, code, content) representing achievement standards
    
    Returns:
        Formatted prompt string for LLM
    """
    prompt = """교과서 내용과 가장 관련 있는 성취기준 번호를 선택하세요.

교과서 내용:
{text}

성취기준:
{candidates}

답변 (번호만):"""
    
    # Format candidates
    candidate_text = "\n".join([
        f"{idx}. {content}"
        for idx, code, content in candidates
    ])
    
    return prompt.format(text=text, candidates=candidate_text)


def create_chat_format_prompt(text: str, candidates: list[tuple[int, str, str]]) -> list[dict]:
    """
    Create a chat-format classification prompt for chat-based models.
    
    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (index, code, content) representing achievement standards
    
    Returns:
        List of chat messages in the format [{"role": "system/user/assistant", "content": "..."}]
    """
    # Format candidates
    candidate_text = "\n".join([
        f"{idx}. [{code}] {content}"
        for idx, code, content in candidates
    ])
    
    messages = [
        {
            "role": "system",
            "content": "당신은 교육과정 전문가입니다. 교과서 텍스트를 분석하여 적합한 교육 성취기준을 선택하는 일을 합니다."
        },
        {
            "role": "user",
            "content": f"""다음 교과서 텍스트를 읽고, 가장 적합한 성취기준의 번호를 하나만 선택해주세요.

# 교과서 텍스트
{text}

# 성취기준 목록
{candidate_text}

가장 적합한 성취기준의 번호만 출력하세요. (예: 1)"""
        }
    ]
    
    return messages


def parse_llm_response(response: str) -> int:
    """
    Parse LLM response to extract the predicted index.
    
    Args:
        response: Raw LLM output string
    
    Returns:
        Predicted index as integer, or -1 if parsing fails
    """
    import re
    
    # Remove whitespace
    response = response.strip()
    
    # Try to find the first number in the response
    match = re.search(r'\b(\d+)\b', response)
    if match:
        return int(match.group(1))
    
    return -1

