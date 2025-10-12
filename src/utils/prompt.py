"""
Prompt templates for LLM-based text classification.
Provides functions to generate prompts for educational content classification.

=== Usage Guide ===

The prompts are structured in 4 sections for easy modification:
    1. System Prompt - Establishes the AI's role + provides achievement standards as knowledge
    2. User Intro - Optional introduction (can be empty)
    3. Content Section - Textbook text to classify
    4. Output Instructions - Specifies the output format

To experiment with different prompts:
    1. Modify the template constants below (SYSTEM_PROMPT, OUTPUT_FORMAT_INSTRUCTION, etc.)
    2. Or pass custom templates to create_classification_prompt()
    
Examples:
    # Default: Output content text
    prompt = create_classification_prompt(text, candidates)
    
    # Alternative: Output code
    prompt = create_classification_prompt(
        text, candidates, 
        output_instruction=OUTPUT_FORMAT_INSTRUCTION_CODE
    )
    
    # Alternative: Output index number
    prompt = create_classification_prompt(
        text, candidates,
        output_instruction=OUTPUT_FORMAT_INSTRUCTION_INDEX
    )
"""

from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Prompt Templates - Separated into 4 sections for easy modification
# ============================================================================

# Default Template (outputs content text)
# Section 1: System Prompt - Establishes the role and context
SYSTEM_PROMPT = """You are an educational curriculum expert. Read the given textbook text and select the most appropriate achievement standard."""

# Section 2: User Prompt Introduction (optional, can be empty)
USER_PROMPT_INTRO = ""

# Section 3: Content Template - Textbook text to classify
# Achievement standards are moved to Section 1 (System Prompt)

# Section 4: Output Format Instructions
OUTPUT_FORMAT_INSTRUCTION = """# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the selected achievement standard content, EXACTLY as written in the list. Do NOT add any explanations, reasoning, or additional text.

Example output format:
친숙한 일반적 주제에 관한 글을 읽고 필자의 의도나 글의 목적을 파악할 수 있다.

# Answer"""


# ============================================================================
# Alternative Templates - Can be easily switched for experimentation
# ============================================================================

# Template variant: Output code instead of content
SYSTEM_PROMPT_CODE = """You are an educational curriculum expert. Read the given textbook text and select the most appropriate achievement standard."""

OUTPUT_FORMAT_INSTRUCTION_CODE = """# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the code of the selected achievement standard. Do NOT add any explanations, reasoning, or additional text.

Example output format:
10영03-04

# Answer"""

# Template variant: Output index number
OUTPUT_FORMAT_INSTRUCTION_INDEX = """# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the number (index) of the selected achievement standard. Do NOT add any explanations, reasoning, or additional text.

Example output format:
5

# Answer"""


class MatchType(Enum):
    """Type of match found when parsing LLM response."""
    EXACT = "exact"  # Exact content match
    PARTIAL = "partial"  # Content partially in response or vice versa
    FUZZY = "fuzzy"  # Similarity-based match
    CODE_PATTERN = "code_pattern"  # Found code pattern in response
    INDEX = "index"  # Found numeric index
    INVALID = "invalid"  # No valid match found


@dataclass
class LLMClassificationResponse:
    """
    Structured response from LLM classification parsing.
    
    Attributes:
        predicted_code: The predicted achievement standard code (e.g., "10영03-04")
        match_type: Type of matching used to extract the prediction
        confidence: Confidence score for fuzzy matches (0.0-1.0), 1.0 for exact matches
        raw_response: Original LLM output text
    """
    predicted_code: str
    match_type: MatchType
    confidence: float
    raw_response: str
    
    @property
    def is_exact_match(self) -> bool:
        """Returns True if the match was exact (not fuzzy or fallback)."""
        return self.match_type == MatchType.EXACT
    
    @property
    def is_valid(self) -> bool:
        """Returns True if a valid prediction was found."""
        return self.match_type != MatchType.INVALID


def create_classification_prompt(
    text: str, 
    candidates: list[tuple[int, str, str]],
    system_prompt: str = None,
    user_intro: str = None,
    output_instruction: str = None
) -> str:
    """
    Create a zero-shot classification prompt for educational content matching.
    
    The prompt is composed of 4 sections:
    1. System prompt: Establishes the role + provides achievement standards as knowledge
    2. User prompt intro: Optional introduction (can be empty)
    3. Content section: Textbook text only
    4. Output format: Instructions on how to format the answer
    
    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (index, code, content) representing achievement standards
        system_prompt: Custom system prompt (uses SYSTEM_PROMPT if None)
        user_intro: Custom user intro (uses USER_PROMPT_INTRO if None)
        output_instruction: Custom output instruction (uses OUTPUT_FORMAT_INSTRUCTION if None)
    
    Returns:
        Formatted prompt string for LLM
    
    Examples:
        # Use default template (outputs content text)
        >>> prompt = create_classification_prompt(text, candidates)
        
        # Use code output template
        >>> prompt = create_classification_prompt(
        ...     text, candidates, 
        ...     output_instruction=OUTPUT_FORMAT_INSTRUCTION_CODE
        ... )
        
        # Use index output template
        >>> prompt = create_classification_prompt(
        ...     text, candidates,
        ...     output_instruction=OUTPUT_FORMAT_INSTRUCTION_INDEX
        ... )
    """
    # Use defaults if not specified
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if user_intro is None:
        user_intro = USER_PROMPT_INTRO
    if output_instruction is None:
        output_instruction = OUTPUT_FORMAT_INSTRUCTION
    
    # Format candidates for system prompt (without code)
    candidate_text = "\n".join([
        f"{idx}. {content}"
        for idx, code, content in candidates
    ])
    
    # Section 1: System prompt with achievement standards
    system_section = (
        f"{system_prompt}\n"
        "\n"
        "# Achievement Standards List\n"
        f"{candidate_text}"
    )
    
    # Section 3: Content section (textbook text only)
    content_section = (
        "# Textbook Text\n"
        f"{text}"
    )
    
    # Combine all sections
    prompt_parts = [
        system_section,
        user_intro,
        content_section,
        output_instruction
    ]
    
    # Filter out empty parts and join with double newlines
    return "\n\n".join(part for part in prompt_parts if part.strip())


def parse_llm_response(response: str, candidates: list[tuple[int, str, str]]) -> LLMClassificationResponse:
    """
    Parse LLM response to extract the predicted achievement standard code.
    
    Args:
        response: Raw LLM output string
        candidates: List of tuples (index, code, content) representing achievement standards
    
    Returns:
        LLMClassificationResponse object containing prediction details
    """
    import re
    from difflib import SequenceMatcher
    
    # Remove whitespace
    response_clean = response.strip()
    
    # Extract codes and contents from candidates
    codes = [code for _, code, _ in candidates]
    contents = [content for _, _, content in candidates]
    
    # Try to find exact content match
    for idx, content in enumerate(contents):
        if content == response_clean:
            return LLMClassificationResponse(
                predicted_code=codes[idx],
                match_type=MatchType.EXACT,
                confidence=1.0,
                raw_response=response
            )
    
    # Try to find partial content match
    for idx, content in enumerate(contents):
        if content in response_clean or response_clean in content:
            return LLMClassificationResponse(
                predicted_code=codes[idx],
                match_type=MatchType.PARTIAL,
                confidence=0.95,
                raw_response=response
            )
    
    # Try fuzzy matching with contents (find best similarity)
    best_match_idx = -1
    best_similarity = 0.0
    similarity_threshold = 0.7  # 70% similarity threshold
    
    for idx, content in enumerate(contents):
        similarity = SequenceMatcher(None, response_clean, content).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_idx = idx
    
    if best_similarity >= similarity_threshold:
        return LLMClassificationResponse(
            predicted_code=codes[best_match_idx],
            match_type=MatchType.FUZZY,
            confidence=best_similarity,
            raw_response=response
        )
    
    # Fallback: try to find code pattern in response (e.g., "10영03-04")
    for code in codes:
        if code in response_clean:
            return LLMClassificationResponse(
                predicted_code=code,
                match_type=MatchType.CODE_PATTERN,
                confidence=0.8,
                raw_response=response
            )
    
    code_pattern = r'(\d+[가-힣]+\d+-\d+)'
    match = re.search(code_pattern, response_clean)
    if match:
        extracted_code = match.group(1)
        if extracted_code in codes:
            return LLMClassificationResponse(
                predicted_code=extracted_code,
                match_type=MatchType.CODE_PATTERN,
                confidence=0.8,
                raw_response=response
            )
    
    # Last fallback: try to find a number and map to index
    match = re.search(r'\b(\d+)\b', response_clean)
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= len(codes):
            return LLMClassificationResponse(
                predicted_code=codes[idx - 1],
                match_type=MatchType.INDEX,
                confidence=0.6,
                raw_response=response
            )
    
    # No valid match found
    return LLMClassificationResponse(
        predicted_code="INVALID",
        match_type=MatchType.INVALID,
        confidence=0.0,
        raw_response=response
    )

