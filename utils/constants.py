"""
Constants for Med-PRM pipeline
===============================
Centralized system prompts and special token definitions.

These are used across multiple scripts:
  - 2_training.py
  - 4_scoring_PRM.py
  - Various analysis scripts
"""

# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================

RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step "
    "of the given explanation. In order to support the evaluation, the relevant documents, "
    "the question, and the explanation are provided sequentially. If the reasoning contains "
    "errors, output - after that step. If the reasoning in a step is logical and valid, "
    "output + after that step."
)
"""
System prompt for RAG-enabled evaluation.
Used when related_docs are available to provide context.
"""

PRM_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step "
    "of the given explanation. In order to support the evaluation, the question and the "
    "explanation are provided. If the reasoning contains errors, output - after that step. "
    "If the reasoning in a step is logical and valid, output + after that step."
)
"""
Standard system prompt for Process Reward Modeling (PRM).
Used for step-by-step reasoning evaluation without RAG.
"""

ORM_SYSTEM_PROMPT = (
    "You are an evaluator assessing the overall quality and correctness of the final answer "
    "in the given explanation. In order to support the evaluation, the question and the "
    "explanation are provided. If the final answer is incorrect or not well-supported, "
    "output -. If the final answer is correct and well-supported, output +."
)
"""
System prompt for Outcome Reward Modeling (ORM).
Used for overall correctness evaluation (not step-wise).
"""

# ==============================================================================
# SPECIAL TOKENS
# ==============================================================================

STEP_MARKER = " ки"  # Special token marking step evaluation points
PLUS_TOKEN = " +"     # Token indicating correct reasoning
MINUS_TOKEN = " -"    # Token indicating incorrect reasoning

# ==============================================================================
# DEFAULTS
# ==============================================================================

DEFAULT_MAX_TOKEN_LEN = 4096
"""Default token budget for full context (question + explanation + docs)."""

DEFAULT_RESERVE_FOR_Q_AND_SOL = 1024
"""Default tokens reserved for question and solution (non-RAG part)."""

DEFAULT_DTYPE = "bfloat16"
"""Default data type for model inference (bfloat16 for A100+, float16 for V100)."""

DEFAULT_ATTENTION_IMPL = "flash_attention_2"
"""Default attention implementation (eager for compatibility)."""

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

MODEL_CONFIGS = {
    "llama-3.1-medprm": {
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "pad_to_multiple_of": 8,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "pad_token": "eos_token",
    },
}
"""Model-specific configurations."""

# ==============================================================================
# DATASET FILTERS
# ==============================================================================

SUPPORTED_DATA_SOURCES = [
    "medqa",
    "mmlu",
    "pubmedqa",
    "med_qa",
    "med_mcqa",
    "usmle",
    "medical",
]
"""Supported medical dataset sources."""

# ==============================================================================
# LABEL TYPES
# ==============================================================================

LABEL_TYPES = {
    "prm_soft_label": "Soft labels for PRM (continuous)",
    "prm_hard_label": "Hard labels for PRM (binary per step)",
    "gemini_label": "Labels from Gemini evaluation",
    "llama_label": "Labels from Llama evaluation",
    "orm_label": "Overall correctness label",
}
"""Supported training label types."""

# ==============================================================================
# INFERENCE OPTIONS
# ==============================================================================

def get_system_prompt(use_rag: bool = False, use_orm: bool = False) -> str:
    """
    Get appropriate system prompt based on configuration.

    Args:
        use_rag: If True, use RAG-enabled prompt
        use_orm: If True, use ORM prompt (only when use_rag=False)

    Returns:
        System prompt string
    """
    if use_rag:
        return RAG_SYSTEM_PROMPT
    elif use_orm:
        return ORM_SYSTEM_PROMPT
    else:
        return PRM_SYSTEM_PROMPT
