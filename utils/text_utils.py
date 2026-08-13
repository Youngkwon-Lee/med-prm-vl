"""
Text processing utilities for Med-PRM
=====================================
Handles question formatting, answer extraction, and text normalization.

Previously duplicated in:
  - python/2_training.py (lines 145-151)
  - python/4_scoring_PRM.py (lines 70-79)
"""

from typing import Any, Dict, List, Optional


def format_question_with_options(item: Dict[str, Any]) -> str:
    """
    Format question with options in standard format.

    Combines question text with multiple choice options in the format:
        "Question text (A) Option 1 (B) Option 2 ..."

    Args:
        item: Dictionary containing:
            - "question": str - main question text
            - "options": List[str] - multiple choice options (optional)

    Returns:
        Formatted question string

    Examples:
        >>> item = {
        ...     "question": "What is 2+2?",
        ...     "options": ["3", "4", "5", "6"]
        ... }
        >>> format_question_with_options(item)
        "What is 2+2? (A) 3 (B) 4 (C) 5 (D) 6"
    """
    q = item.get("question", "")
    opts = item.get("options", [])

    if not opts:
        return q

    formatted = "".join(f" ({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(opts))
    return q + formatted


def extract_answer_from_solution(
    solution_text: str,
    answer_format: str = "last_line"
) -> str:
    """
    Extract the final answer from a solution text.

    Supports multiple extraction strategies.

    Args:
        solution_text: Full solution text
        answer_format: Strategy for extraction
            - "last_line": Take last non-empty line
            - "answer_statement": Look for "Answer:" or "Final:" prefix
            - "final_answer_box": Look for boxed answer

    Returns:
        Extracted answer string or empty string if not found
    """
    if not solution_text or not isinstance(solution_text, str):
        return ""

    if answer_format == "answer_statement":
        lines = solution_text.strip().split('\n')
        for line in reversed(lines):
            if "answer:" in line.lower() or "final:" in line.lower():
                return line.split(":", 1)[-1].strip()
        return ""

    elif answer_format == "final_answer_box":
        if "\\boxed{" in solution_text:
            start = solution_text.find("\\boxed{") + 7
            end = solution_text.find("}", start)
            if end > start:
                return solution_text[start:end].strip()
        return ""

    elif answer_format == "last_line":
        lines = [line.strip() for line in solution_text.strip().split('\n') if line.strip()]
        return lines[-1] if lines else ""

    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Converts to lowercase, removes extra whitespace, and strips punctuation.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string

    Examples:
        >>> normalize_answer("The answer is (C) Correct .")
        "the answer is (c) correct"
    """
    if not answer:
        return ""

    # Convert to lowercase
    normalized = answer.lower()

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    # Keep only alphanumeric and spaces (for medical terms, keep letters/numbers)
    # But preserve some punctuation like hyphen and parentheses
    import re
    normalized = re.sub(r'[^\w\s\-\(\)]', '', normalized)

    return normalized.strip()


def split_solution_into_steps(
    solution_text: str,
    step_marker: str = "Step ",
    bullet_markers: Optional[List[str]] = None
) -> List[str]:
    """
    Split solution into individual steps for step-wise evaluation.

    Args:
        solution_text: Full solution text
        step_marker: Prefix indicating step start (e.g., "Step ", "1. ")
        bullet_markers: Additional markers for steps (e.g., ["-", "*", "•"])

    Returns:
        List of step texts

    Examples:
        >>> solution = '''Step 1: Read the question
        ... Step 2: Identify key terms
        ... Step 3: Formulate answer'''
        >>> steps = split_solution_into_steps(solution)
        >>> len(steps)
        3
    """
    if not solution_text:
        return []

    lines = solution_text.split('\n')
    steps = []
    current_step = []

    if bullet_markers is None:
        bullet_markers = ['-', '*', '•', '→']

    for line in lines:
        stripped = line.strip()

        # Check if line starts a new step
        is_new_step = (
            stripped.startswith(step_marker) or
            any(stripped.startswith(f"{marker} ") for marker in bullet_markers) or
            (len(stripped) > 0 and stripped[0].isdigit() and ". " in stripped[:3])
        )

        if is_new_step and current_step:
            steps.append('\n'.join(current_step).strip())
            current_step = [stripped]
        else:
            current_step.append(stripped)

    if current_step:
        steps.append('\n'.join(current_step).strip())

    return [s for s in steps if s]  # Remove empty steps


def create_evaluation_prompt(
    question: str,
    explanation: str,
    documents: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tokenizer: Optional[Any] = None
) -> str:
    """
    Create evaluation prompt from components.

    Assembles question, documents, and explanation into standardized format.

    Args:
        question: Formatted question text
        explanation: Solution explanation
        documents: Optional document block from RAG
        system_prompt: System instruction (for context only)
        tokenizer: Tokenizer for chat template (if not provided, returns plain text)

    Returns:
        Chat-templated prompt or plain text format
    """
    # Build content
    content_parts = []

    if documents:
        content_parts.append(documents)

    content_parts.append(f"Question: {question}\n\nExplanation: {explanation}")
    content = "".join(content_parts)

    # If tokenizer provided, use chat template
    if tokenizer:
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Otherwise return plain format
    return f"System: {system_prompt}\n\nUser: {content}"


def truncate_text(text: str, max_chars: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum character count.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text

    return text[:max_chars - len(suffix)] + suffix
