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

import json
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Prompt Templates - Separated into 4 sections for easy modification
# ============================================================================

# Default Template (outputs content text)
# Section 1: System Prompt
SYSTEM_PROMPT_CODE = """You are an educational curriculum expert. Your task is to match textbook content with achievement standards.

WHAT ARE ACHIEVEMENT STANDARDS:
Achievement standards are specific learning objectives that define what students should know and be able to do at a particular grade level. Each standard describes:
- The specific knowledge or skills students need to acquire
- The level of understanding or performance expected
- The context or situation where learning should be applied

HOW TO MATCH TEXTBOOK CONTENT TO STANDARDS:
1. Read the textbook text carefully and identify its primary educational purpose
2. Ask yourself: "What is this content trying to teach students?"
3. Look for key indicators:
   - What subject knowledge is being presented?
   - What skills or abilities are students expected to develop?
   - What cognitive processes are involved (understanding, applying, analyzing)?
4. Select the standard that most directly aligns with the main learning goal

IMPORTANT PRINCIPLES:
- Focus on the PRIMARY learning objective, not secondary or supporting content
- Consider what students should be able to DO after studying this content
- Match based on educational intent, not just topic similarity"""
# Section 2: User Prompt Introduction (optional, can be empty)
USER_PROMPT_INTRO = ""

# Section 3: Content Template - Textbook text to classify
# Achievement standards are moved to Section 1 (System Prompt)

# Section 4: Output Format Instructions
OUTPUT_FORMAT_INSTRUCTION_CODE = """# Task
Analyze the textbook text and select the ONE achievement standard that best matches its primary educational objective.

# Output Format
Output ONLY the achievement standard code. No explanations, no additional text.

Correct format:
10영03-04

Wrong formats:
❌ "10영03-04 because..."
❌ Code: 10영03-04

# Answer"""

OUTPUT_FORMAT_INSTRUCTION_FEW_SHOT_CODE = """# Task
Review the example patterns shown in the "Few-Shot Examples" section above. Each example demonstrates how a textbook text was matched to its corresponding achievement standard.

Apply the same analysis process to classify the "Textbook Text" provided above.

# Output Format
Output ONLY the achievement standard code. No explanations, no additional text.

Correct format:
10영03-04

# Answer"""


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
    subject: str, num_examples: int = 3, file_name: str = "few_shot_examples.json"
) -> str:
    """
    Load few-shot examples from a JSON file.
    """
    with open(file_name, "r") as f:
        data = json.load(f)
    examples = data[subject][:num_examples]
    examples_str = "\n".join(
        [f"{idx}. {example['text']}" for idx, example in enumerate(examples)]
    )
    return examples_str


def create_classification_prompt(
    text: str,
    candidates: list[tuple[int, str, str]],
    system_prompt: str = None,
    user_intro: str = None,
    output_instruction: str = None,
    few_shot: bool = False,
    file_name: str = "few_shot_examples.json",
    subject: str = None,
    num_examples: int = 3,
) -> str:
    """
    Create a classification prompt for educational content matching.

    The prompt is composed of 4 sections:
    1. System prompt: Establishes the role + provides achievement standards as knowledge
    2. User prompt intro: Optional introduction (can be empty)
    3. Content section: Textbook text only
    4. Output format: Instructions on how to format the answer

    If few_shot is True, the prompt will include a few-shot example.

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
    if few_shot:
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_CODE
        if user_intro is None:
            user_intro = USER_PROMPT_INTRO
        if output_instruction is None:
            output_instruction = OUTPUT_FORMAT_INSTRUCTION_FEW_SHOT_CODE
    else:
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_CODE
        if user_intro is None:
            user_intro = USER_PROMPT_INTRO
        if output_instruction is None:
            output_instruction = OUTPUT_FORMAT_INSTRUCTION_CODE

    # Format candidates for system prompt (without code)
    candidate_text = "\n".join(
        [f"{code}: {content}" for idx, code, content in candidates]
    )

    # Section 1: System prompt with achievement standards
    system_section = (
        f"{system_prompt}\n" "\n" "# Achievement Standards List\n" f"{candidate_text}"
    )
    if few_shot:
        few_shot_examples = load_few_shot_examples(subject, num_examples, file_name)
        system_section = (
            system_section + "\n" + "# Few-Shot Examples\n" + few_shot_examples
        )

    # Section 3: Content section (textbook text only)
    content_section = "# Textbook Text\n" f"{text}"

    # Combine all sections
    prompt_parts = [system_section, user_intro, content_section, output_instruction]

    # Filter out empty parts and join with double newlines
    return "\n\n".join(part for part in prompt_parts if part.strip())


def parse_llm_response(
    response: str, candidates: list[tuple[int, str, str]]
) -> LLMClassificationResponse:
    """
    Parse LLM response to extract the predicted achievement standard code.

    The LLM now directly outputs the code (e.g., "10영03-04") instead of the content.
    This simplifies the parsing logic significantly.

    Args:
        response: Raw LLM output string (expected to be a code like "10영03-04")
        candidates: List of tuples (index, code, content) representing achievement standards

    Returns:
        LLMClassificationResponse object containing prediction details
    """
    # Remove whitespace
    response_clean = response.strip()

    # Extract codes from candidates
    codes = [code for _, code, _ in candidates]

    # Try to find exact code match
    if response_clean in codes:
        return LLMClassificationResponse(
            predicted_code=response_clean,
            match_type=MatchType.EXACT,
            confidence=1.0,
            raw_response=response,
        )

    # Try to find partial code match (code in response or response in code)
    for code in codes:
        if code in response_clean or response_clean in code:
            return LLMClassificationResponse(
                predicted_code=code,
                match_type=MatchType.PARTIAL,
                confidence=0.8,
                raw_response=response,
            )

    # No valid match found
    return LLMClassificationResponse(
        predicted_code="INVALID",
        match_type=MatchType.INVALID,
        confidence=0.0,
        raw_response=response,
    )
