"""
Agentic prompt templates for LLM-based text classification.
Provides functions to generate prompts for educational content classification using agentic workflow.

=== Agentic Workflow ===

Step 1: Query Generation
    - Analyze textbook text
    - Generate a descriptive query
    - Use infer_top_k tool to retrieve candidate achievement standards

Step 2: Final Selection
    - Review candidates from Step 1
    - Select the best matching achievement standard code

=== Usage Guide ===

For Step 1 (Query Generation):
    prompt = create_agentic_step1_prompt(text)
    # LLM generates query and calls infer_top_k tool
    
For Step 2 (Final Selection):
    prompt = create_agentic_step2_prompt(text, candidates)
    # LLM selects the best code from candidates
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ============================================================================
# Agentic System Prompts
# ============================================================================

AGENTIC_SYSTEM_PROMPT_STEP1 = """You are an educational curriculum expert specializing in analyzing textbook content.

Your role is to analyze textbook text and generate a descriptive query that captures the key educational concepts and learning objectives present in the text.

This query will be used to search for matching achievement standards in a database.

IMPORTANT CAPABILITIES:
You have access to a tool called 'infer_top_k' that can search for achievement standards similar to a given query."""

AGENTIC_SYSTEM_PROMPT_STEP2 = """You are an educational curriculum expert. Your task is to select the most appropriate achievement standard from a list of candidates.

WHAT ARE ACHIEVEMENT STANDARDS:
Achievement standards are specific learning objectives that each standard describes:
- The specific knowledge or skills students need to acquire
- The level of understanding or performance expected
- The context or situation where learning should be applied

HOW TO MATCH TEXTBOOK CONTENT TO STANDARDS:
1. Read the textbook text carefully and determine the subject area
2. Identify the primary educational purpose of the text
3. Select the standard that most directly aligns with the main learning goal"""

# ============================================================================
# Agentic Output Format Instructions
# ============================================================================

# Step 1: Query Generation
AGENTIC_OUTPUT_FORMAT_STEP1 = """# Task
Analyze the textbook text below and generate a search query that describes the achievement standard that would match this text.

Your query should:
- Capture the main subject area and learning objective
- Be concise but descriptive (1-2 sentences)
- Focus on the educational purpose rather than specific details

# Instructions
1. Read and analyze the textbook text carefully
2. Identify the core educational concept being taught
3. Generate a descriptive query for searching

# Correct format:
일차방정식의 풀이 방법 이해 활용

# Query"""

AGENTIC_OUTPUT_FORMAT_STEP1_FEW_SHOT = """# Task
Review the example patterns shown in the "Few-Shot Examples" section above. Each example demonstrates how a textbook text was analyzed to generate an effective search query.

Apply the same analysis process to generate a search query for the "Textbook Text" provided above.

Your query should:
- Capture the main subject area and learning objective
- Be concise but descriptive (1-2 sentences)
- Focus on the educational purpose rather than specific details

# Instructions
Based on the patterns learned from the examples:
1. Read and analyze the textbook text carefully
2. Identify the core educational concept being taught
3. Generate a descriptive query similar to the example patterns

# Correct format:
일차방정식의 풀이 방법 이해 활용

# Query"""

# Step 2: Final Selection
AGENTIC_OUTPUT_FORMAT_STEP2 = """#  Task
Analyze the textbook text and select the ONE achievement standard that best matches its primary educational objective.

# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the achievement standard code. Do NOT add any explanations, reasoning, or additional text.

Correct format:
10영03-04

# Answer"""

AGENTIC_OUTPUT_FORMAT_STEP2_FEW_SHOT = """# Task
Review the example patterns shown in the "Few-Shot Examples" section above. Each example demonstrates how a textbook text was matched to its corresponding achievement standard.

Apply the same analysis process to classify the "Textbook Text" provided above.

# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the achievement standard code. Do NOT add any explanations, reasoning, or additional text.

Correct format:
10영03-04

# Answer"""

# ============================================================================
# Tool Descriptions
# ============================================================================

TOOL_DESCRIPTION_INFER_TOP_K = """Tool Name: infer_top_k

Description: Searches for achievement standards that are similar to the given query.

Input: A descriptive query about the educational content

Output Format:
1: [code]; [content]
2: [code]; [content]
3: [code]; [content]
...

The results are sorted from most similar to least similar.

Example:
Query: '일차방정식의 풀이 방법 이해 활용'
Output:
1: 09수04-02; 일차방정식의 풀이 방법을 이해하고 활용할 수 있다
2: 09수04-03; 연립일차방정식의 풀이 방법을 이해하고 활용할 수 있다
3: 09수03-01; 문자를 사용한 식을 이해하고 간단한 식을 표현할 수 있다"""


class MatchType(Enum):
    """Type of match found when parsing LLM response."""

    EXACT = "exact"  # Exact code match
    PARTIAL = "partial"  # Code partially in response or vice versa
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


def load_few_shot_examples(
    subject: str,
    num_examples: int = 5,
    file_path: str | Path | None = None,
) -> str:
    """
    Load few-shot examples from a JSON file.

    Args:
        subject: Subject name (e.g., "english", "math")
        num_examples: Number of examples to load
        file_path: Custom file path (optional)

    Returns:
        Formatted few-shot examples string
    """
    if file_path is None:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        few_shot_file = (
            PROJECT_ROOT / "dataset" / "few_shot_examples" / f"{subject}.json"
        )
    else:
        few_shot_file = Path(file_path / f"{subject}.json")

    with open(few_shot_file, "r") as f:
        data = json.load(f)
    examples = data[:num_examples]
    examples_list = []
    for idx, example in enumerate(examples, 1):
        examples_list.append(
            f"Example {idx}:\n"
            f"Text: {example['text']}\n"
            f"Achievement Standard: {example['content']}\n"
            f"Answer code: {example['code']}"
        )
    return "\n\n".join(examples_list)


# ============================================================================
# Utility Functions
# ============================================================================


def format_candidates_for_step2(candidates: list[tuple[int, str, str]]) -> str:
    """
    Format candidates for Step 2 prompt.

    Args:
        candidates: List of tuples (rank, code, content) from infer_top_k

    Returns:
        Formatted string with numbered candidates

    Example:
        >>> candidates = [(1, "10영03-04", "영어로 의견 표현하기"), (2, "10영03-05", "...")]
        >>> format_candidates_for_step2(candidates)
        "1: 10영03-04; 영어로 의견 표현하기\\n2: 10영03-05; ..."
    """
    return "\n".join(
        [f"{rank}: {code}; {content}" for rank, code, content in candidates]
    )


# ============================================================================
# Agentic Step 2: Final Selection
# ============================================================================


def create_rag_chat_prompt(
    text: str,
    candidates: list[tuple[int, str, str]],
    completion: str = "",
    system_prompt: str | None = None,
    output_instruction: str | None = None,
    for_inference: bool = False,
    few_shot: bool = False,
    subject: str | None = None,
    num_examples: int = 5,
    few_shot_file: str | Path | None = None,
) -> dict:
    """
    Create a chat-based prompt for Step 2: Final selection.

    Returns a dictionary with 'messages' field for chat-based models.

    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (rank, code, content) from infer_top_k
        completion: The achievement standard code (answer) for training
        system_prompt: Custom system prompt (uses AGENTIC_SYSTEM_PROMPT_STEP2 if None)
        output_instruction: Custom output instruction (uses default if None)
        for_inference: If True, exclude assistant message
        few_shot: Whether to include few-shot examples
        subject: Subject name for few-shot examples
        num_examples: Number of few-shot examples
        few_shot_file: Custom few-shot file path

    Returns:
        Dictionary with 'messages' field containing role-based messages
    """
    if few_shot:
        if system_prompt is None:
            system_prompt = AGENTIC_SYSTEM_PROMPT_STEP2
        if output_instruction is None:
            output_instruction = AGENTIC_OUTPUT_FORMAT_STEP2_FEW_SHOT
    else:
        if system_prompt is None:
            system_prompt = AGENTIC_SYSTEM_PROMPT_STEP2
        if output_instruction is None:
            output_instruction = AGENTIC_OUTPUT_FORMAT_STEP2

    # Format candidates for system prompt
    # different from prompt.py
    # because we use rank instead of index
    candidate_text_with_rank = format_candidates_for_step2(candidates)

    # System message: Role definition + Achievement Standards
    system_content = (
        f"{system_prompt}\n"
        "\n"
        "# Achievement Standards List\n"
        "The infer_top_k tool has returned the following candidates (ranked by similarity):\n"
        f"{candidate_text_with_rank}"
    )

    # Add few-shot examples if requested
    if few_shot:
        few_shot_examples = load_few_shot_examples(
            subject=subject,
            num_examples=num_examples,
            file_path=few_shot_file,
        )
        system_content = (
            f"{system_content}\n" "# Few-Shot Examples\n" f"{few_shot_examples}"
        )

    # User message: Textbook text + Output instructions
    user_content = "# Textbook Text\n" f"{text}\n" "\n" f"{output_instruction}"

    # Build messages list
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Add assistant message only for training (not inference)
    if not for_inference:
        messages.append({"role": "assistant", "content": completion})

    return {"messages": messages}


# ============================================================================
# Response Parsing
# ============================================================================


def parse_llm_response(
    response: str, candidates: list[tuple[int, str, str]]
) -> LLMClassificationResponse:
    """
    Parse LLM response to extract the predicted achievement standard code.

    The LLM directly outputs the code (e.g., "10영03-04").

    Args:
        response: Raw LLM output string (expected to be a code like "10영03-04")
        candidates: List of tuples (rank, code, content) representing achievement standards

    Returns:
        LLMClassificationResponse object containing prediction details
    """
    # Remove whitespace
    response_clean = response.strip()

    # Extract codes from candidates
    codes = [code for _, code, _ in candidates]

    # Strategy 1: Exact code match
    if response_clean in codes:
        return LLMClassificationResponse(
            predicted_code=response_clean,
            match_type=MatchType.EXACT,
            confidence=1.0,
            raw_response=response,
        )

    # Strategy 2: Partial code match (code in response or response in code)
    for code in codes:
        if code in response_clean or response_clean in code:
            return LLMClassificationResponse(
                predicted_code=code,
                match_type=MatchType.PARTIAL,
                confidence=0.8,
                raw_response=response,
            )

    # Strategy 3: No valid match found
    return LLMClassificationResponse(
        predicted_code="INVALID",
        match_type=MatchType.INVALID,
        confidence=0.0,
        raw_response=response,
    )
