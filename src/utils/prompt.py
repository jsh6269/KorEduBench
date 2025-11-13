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
from pathlib import Path

# ============================================================================
# Prompt Templates - Separated into 4 sections for easy modification
# ============================================================================

# Default Template (outputs content text)
# Section 1: System Prompt - Establishes the role and context
SYSTEM_PROMPT_CODE = """You are an educational curriculum expert. Your task is to match textbook text with the most appropriat achievement standards.

WHAT ARE ACHIEVEMENT STANDARDS:
Achievement standards are specific learning objectives that each standard describes:
- The specific knowledge or skills students need to acquire
- The level of understanding or performance expected
- The context or situation where learning should be applied

HOW TO MATCH TEXTBOOK CONTENT TO STANDARDS:
1. Read the textbook text carefully and identify its primary educational purpose
2. Select the standard that most directly aligns with the main learning goal"""

# Section 2: User Prompt Introduction (optional, can be empty)
USER_PROMPT_INTRO = ""

# Section 3: Content Template - Textbook text to classify
# Achievement standards are moved to Section 1 (System Prompt)

# Section 4: Output Format Instructions
OUTPUT_FORMAT_INSTRUCTION_CODE = """# Task
Analyze the textbook text and select the ONE achievement standard that best matches its primary educational objective.

# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the index number of the selected achievement standard. Do NOT add any explanations, reasoning, or additional text.

Correct output format:
15

# Answer"""

OUTPUT_FORMAT_INSTRUCTION_FEW_SHOT_CODE = """# Task
Review the example patterns shown in the "Few-Shot Examples" section above. Each example demonstrates how a textbook text was matched to its corresponding achievement standard.

Apply the same analysis process to classify the "Textbook Text" provided above.

# Instructions
Select ONLY ONE achievement standard that best describes the textbook text above.

IMPORTANT: Output ONLY the index number of the selected achievement standard. Do NOT add any explanations, reasoning, or additional text.

Correct output format:
5

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
    subject: str, num_examples: int = 5
) -> str:
    """
    Load few-shot examples from a JSON file.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    few_shot_file = PROJECT_ROOT / "dataset" / "few_shot_examples" / f"{subject}.json"
    
    with open(few_shot_file, "r") as f:
        data = json.load(f)
    examples = data[:num_examples]
    examples_list = []
    for idx, example in enumerate(examples, 1):
        examples_list.append(
            f"Example {idx}:\n"
            f"Text: {example['text']}\n"
            f"Achievement Standard: {example['content']}" 
        )    
    return "\n\n".join(examples_list)


def create_classification_prompt(
    text: str,
    candidates: list[tuple[int, str, str]],
    system_prompt: str = None,
    user_intro: str = None,
    output_instruction: str = None,
    few_shot: bool = False,
    subject: str = None,
    num_examples: int = 5,
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
        [f"{idx}: {content}" for idx, code, content in candidates]
    ) 

    # Section 1: System prompt with achievement standards
    system_section = (
        f"{system_prompt}\n" "\n" "# Achievement Standards List\n" f"{candidate_text}"
    )
    if few_shot:
        few_shot_examples = load_few_shot_examples(subject, num_examples)
        system_section = (
            system_section + "\n" + "# Few-Shot Examples\n" + few_shot_examples
        )

    # Section 3: Content section (textbook text only)
    content_section = "# Textbook Text\n" f"{text}"

    # Combine all sections
    prompt_parts = [system_section, user_intro, content_section, output_instruction]

    # Filter out empty parts and join with double newlines
    return "\n\n".join(part for part in prompt_parts if part.strip())


def create_chat_classification_prompt(
    text: str,
    candidates: list[tuple[int, str, str]],
    completion: str,
    system_prompt: str = None,
    output_instruction: str = None,
    for_inference: bool = False,
) -> dict:
    """
    Create a chat-based classification prompt for training or inference with message roles.

    Returns a dictionary in the format expected by SFTTrainer with a 'messages' field.

    Args:
        text: The textbook excerpt to classify
        candidates: List of tuples (index, code, content) representing achievement standards
        completion: The achievement standard code (answer) for the assistant role
        system_prompt: Custom system prompt (uses SYSTEM_PROMPT_CODE if None)
        output_instruction: Custom output instruction (uses OUTPUT_FORMAT_INSTRUCTION_CODE if None)
        for_inference: If True, exclude assistant message (for inference mode)

    Returns:
        Dictionary with 'messages' field containing list of role-based messages

    Example:
        >>> # Training mode
        >>> result = create_chat_classification_prompt(text, candidates, "10영03-04")
        >>> result.keys()
        dict_keys(['messages'])
        >>> # Inference mode
        >>> result = create_chat_classification_prompt(text, candidates, "", for_inference=True)
    """
    # Use defaults if not specified
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_CODE
    if output_instruction is None:
        output_instruction = OUTPUT_FORMAT_INSTRUCTION_CODE

    # Format candidates for system prompt
    candidate_text = "\n".join(
    [f"{idx}: {content}" for idx, code, content in candidates]
    )

    # System message: Role definition + Achievement Standards
    system_content = (
        f"{system_prompt}\n" "\n" "# Achievement Standards List\n" f"{candidate_text}"
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

    # Strategy 1: Exact match
    try:
        predicted_idx = int(response_clean)
        
        # IndexError check
        if 0 <= predicted_idx < len(candidates):
            _, code, _ = candidates[predicted_idx]
            
            return LLMClassificationResponse(
                predicted_code=code,
                match_type=MatchType.EXACT,
                confidence=1.0,
                raw_response=response,
            )
    except ValueError:
        pass  # Out of range → Strategy 2
    
    # Strategy 2: Find numbers in the response
    import re
    numbers = re.findall(r'\b\d+\b', response_clean)
    
    for num_str in numbers:
        try:
            idx = int(num_str)
            
            # IndexError check
            if 0 <= idx < len(candidates):
                _, code, _ = candidates[idx]
                
                return LLMClassificationResponse(
                    predicted_code=code,
                    match_type=MatchType.PARTIAL,
                    confidence=0.8,
                    raw_response=response,
                )
        except ValueError:
            continue
    
    # Strategy 3: Not found
    return LLMClassificationResponse(
        predicted_code="",
        match_type=MatchType.INVALID,
        confidence=0.0,
        raw_response=response,
    )
