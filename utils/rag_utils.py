"""
RAG (Retrieval-Augmented Generation) utilities for Med-PRM
===========================================================
Handles document processing and token budget management for RAG mode.

Previously duplicated in:
  - python/2_training.py (lines 127-143)
  - python/4_scoring_PRM.py (lines 82-96)
"""

from typing import Any, Dict, List, Optional, Tuple
from .constants import DEFAULT_MAX_TOKEN_LEN, DEFAULT_RESERVE_FOR_Q_AND_SOL


def truncate_related_docs(
    docs: List[str],
    tokenizer: Any,
    max_total_len: int = DEFAULT_MAX_TOKEN_LEN,
    reserve_for_q_and_sol: int = DEFAULT_RESERVE_FOR_Q_AND_SOL
) -> List[str]:
    """
    Truncate related documents to fit within token budget.

    Greedily includes documents until token budget is exhausted.
    Preserves document order and completeness (doesn't split documents).

    Args:
        docs: List of document strings
        tokenizer: HuggingFace tokenizer with __call__ method
        max_total_len: Total token budget for full context
        reserve_for_q_and_sol: Tokens reserved for question and solution

    Returns:
        List of documents that fit within budget

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> docs = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
        >>> kept_docs = truncate_related_docs(docs, tokenizer, 4096, 1024)
        >>> len(kept_docs) <= len(docs)
        True
    """
    if not docs or not isinstance(docs, list):
        return []

    kept = []
    used = 0
    budget = max_total_len - reserve_for_q_and_sol

    for doc in docs:
        if not isinstance(doc, str):
            continue

        # Tokenize document
        tokens = tokenizer(doc, add_special_tokens=False)["input_ids"]
        doc_token_len = len(tokens)

        # Check if adding this document exceeds budget
        # +1 for separator token between documents
        if used + doc_token_len + 1 > budget:
            break

        kept.append(doc)
        used += doc_token_len + 1

    return kept


def filter_documents_by_relevance(
    docs: List[Dict[str, Any]],
    score_threshold: float = 0.5,
    score_key: str = "relevance_score"
) -> List[str]:
    """
    Filter documents by relevance score.

    Args:
        docs: List of document dictionaries (with content and score)
        score_threshold: Minimum relevance score
        score_key: Key containing relevance score

    Returns:
        List of document content strings above threshold
    """
    filtered = []

    for doc in docs:
        if isinstance(doc, dict):
            score = doc.get(score_key, 1.0)
            if score >= score_threshold and "content" in doc:
                filtered.append(doc["content"])
        elif isinstance(doc, str):
            filtered.append(doc)

    return filtered


def format_document_block(
    docs: List[str],
    format_style: str = "default"
) -> str:
    """
    Format list of documents into a text block for input.

    Args:
        docs: List of document strings
        format_style: Formatting style
            - "default": "Document 1: ...\n\nDocument 2: ..."
            - "citations": "[1] ... [2] ..."
            - "separator": "=== DOC 1 ===\n..." (more explicit)

    Returns:
        Formatted document block string

    Examples:
        >>> docs = ["First doc", "Second doc"]
        >>> block = format_document_block(docs, "default")
        >>> "Document 1:" in block
        True
    """
    if not docs:
        return ""

    if format_style == "citations":
        return "".join(f"[{i+1}] {doc}\n\n" for i, doc in enumerate(docs))

    elif format_style == "separator":
        return "".join(f"=== DOCUMENT {i+1} ===\n{doc}\n\n" for i, doc in enumerate(docs))

    else:  # default
        return "".join(f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs))


def estimate_token_usage(
    question: str,
    explanation: str,
    docs: Optional[List[str]] = None,
    tokenizer: Optional[Any] = None
) -> Dict[str, int]:
    """
    Estimate token usage for a complete prompt.

    Useful for debugging and understanding token budget constraints.

    Args:
        question: Question text
        explanation: Explanation text
        docs: Optional list of documents
        tokenizer: Tokenizer for accurate token count

    Returns:
        Dictionary with token counts for each component

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> q = "What is the diagnosis?"
        >>> e = "The patient presented with..."
        >>> d = ["Document 1", "Document 2"]
        >>> usage = estimate_token_usage(q, e, d, tokenizer)
        >>> usage["question"] > 0
        True
    """
    usage = {}

    if tokenizer:
        usage["question"] = len(tokenizer(question, add_special_tokens=False)["input_ids"])
        usage["explanation"] = len(tokenizer(explanation, add_special_tokens=False)["input_ids"])
        usage["documents"] = sum(
            len(tokenizer(doc, add_special_tokens=False)["input_ids"]) + 1
            for doc in (docs or [])
        )
        usage["overhead"] = 50  # Rough estimate for system prompt and special tokens
        usage["total"] = sum(usage.values())
    else:
        # Rough approximation: 1 token ≈ 4 characters
        usage["question"] = len(question) // 4
        usage["explanation"] = len(explanation) // 4
        usage["documents"] = sum(len(doc) // 4 + 1 for doc in (docs or []))
        usage["overhead"] = 50
        usage["total"] = sum(usage.values())

    return usage


def validate_rag_config(
    max_total_len: int,
    reserve_for_q_and_sol: int
) -> Tuple[bool, str]:
    """
    Validate RAG configuration parameters.

    Args:
        max_total_len: Total token budget
        reserve_for_q_and_sol: Reserved tokens for Q&A

    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_total_len <= 0:
        return False, "max_total_len must be positive"

    if reserve_for_q_and_sol <= 0:
        return False, "reserve_for_q_and_sol must be positive"

    if reserve_for_q_and_sol >= max_total_len:
        return False, "reserve_for_q_and_sol must be less than max_total_len"

    doc_budget = max_total_len - reserve_for_q_and_sol
    if doc_budget < 256:
        return False, f"Document budget too small ({doc_budget} tokens). Increase max_total_len."

    return True, ""
