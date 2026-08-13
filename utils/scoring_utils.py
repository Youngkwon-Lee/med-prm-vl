"""
PRM scoring utilities for Med-PRM
=================================
Core scoring functions for Process Reward Modeling evaluation.

Centralizes scoring logic from:
  - python/4_scoring_PRM.py (lines 162-192)
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn.functional import softmax

from .constants import STEP_MARKER, PLUS_TOKEN, MINUS_TOKEN


def get_prm_scores(
    model: Any,
    tokenizer: Any,
    text: str,
    plus_id: int,
    minus_id: int,
    step_marker: str = STEP_MARKER,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate PRM scores for a given text.

    Evaluates the probability of '+' (correct) vs '-' (incorrect) token
    at each step marker position in the text.

    Args:
        model: Language model
        tokenizer: Tokenizer for the model
        text: Input text with step markers
        plus_id: Token ID for '+' token
        minus_id: Token ID for '-' token
        step_marker: Special token marking steps (default: " ки")
        device: Device to run on (None = use model device)

    Returns:
        Dictionary containing:
            - "plus_probs": List of plus probabilities at each step
            - "min_plus_prob": Minimum probability across steps
            - "final_plus_prob": Probability at final step

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("...")
        >>> plus_id, minus_id = get_plus_minus_ids(tokenizer)
        >>> text = "Reasoning ки Correct ки Another step ки"
        >>> scores = get_prm_scores(model, tokenizer, text, plus_id, minus_id)
        >>> scores["final_plus_prob"]
        0.85
    """
    if device is None:
        device = model.device
    if (
        not isinstance(plus_id, int)
        or isinstance(plus_id, bool)
        or not isinstance(minus_id, int)
        or isinstance(minus_id, bool)
    ):
        raise TypeError(
            f"Token IDs must be integers: plus_id={plus_id}, minus_id={minus_id}"
        )

    # Encode text with offset mapping to find step positions
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offset_mapping"][0]

    # Get model logits
    with torch.no_grad():
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).logits[0]  # Shape: [seq_len, vocab_size]
    vocab_size = logits.shape[-1]
    for name, token_id in (("plus_id", plus_id), ("minus_id", minus_id)):
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"{name} {token_id} out of bounds for vocab_size {vocab_size}"
            )


    # Find positions of step marker
    positions = []
    for i, (s, e) in enumerate(offsets):
        if text[s:e] == step_marker:
            positions.append(i)

    # Calculate probabilities at each step
    plus_probs = []
    for pos in positions:
        if pos >= logits.size(0):
            continue

        # Get logits for '+' and '-' tokens at this position
        two_logits = torch.stack([logits[pos][plus_id], logits[pos][minus_id]])

        # Compute softmax probabilities
        probs = softmax(two_logits, dim=0)
        plus_prob = probs[0].item()
        plus_probs.append(plus_prob)

    # Compute aggregated scores
    min_plus_prob = min(plus_probs) if plus_probs else None
    final_plus_prob = plus_probs[-1] if plus_probs else None

    return {
        "plus_probs": plus_probs,
        "min_plus_prob": min_plus_prob,
        "final_plus_prob": final_plus_prob,
    }


def batch_get_prm_scores(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    plus_id: int,
    minus_id: int,
    step_marker: str = STEP_MARKER,
    batch_size: int = 8,
    device: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Calculate PRM scores for multiple texts in batch.

    Processes texts in batches for better GPU utilization.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        plus_id: Token ID for '+'
        minus_id: Token ID for '-'
        step_marker: Step marker token
        batch_size: Number of texts to process in parallel
        device: Device to run on

    Returns:
        List of score dictionaries

    Note:
        This requires padding and careful attention masking.
        For now, this is a placeholder that processes individually.
        True batch processing would require custom batching logic.
    """
    results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        for text in batch_texts:
            scores = get_prm_scores(
                model,
                tokenizer,
                text,
                plus_id,
                minus_id,
                step_marker,
                device
            )
            results.append(scores)

    return results


def select_best_solution(
    solutions: List[Dict[str, Any]],
    selection_method: str = "processbench"
) -> Optional[Dict[str, Any]]:
    """
    Select best solution based on PRM scores.

    Args:
        solutions: List of solution dicts with PRM scores
        selection_method: Selection strategy
            - "processbench": Use min score (lowest bottleneck step)
            - "bon": Use final score
            - "harmonic_mean": Use harmonic mean of all scores

    Returns:
        Best solution dict or None if no valid solutions
    """
    # Filter solutions with valid scores
    valid = [s for s in solutions if s.get("PRM_score") not in (None, float("-inf"))]

    if not valid:
        return None

    if selection_method == "processbench":
        # ProcessBench: Select path with highest min score
        return max(valid, key=lambda s: s.get("PRM_min_score", float("-inf")))

    elif selection_method == "bon":
        # Best-of-N: Select path with highest final score
        return max(valid, key=lambda s: s.get("PRM_score", float("-inf")))

    elif selection_method == "harmonic_mean":
        # Use harmonic mean of all step scores
        def harmonic_mean(s):
            scores = s.get("PRM_score_list", [])
            if not scores or any(x <= 0 for x in scores):
                return 0
            return len(scores) / sum(1/x for x in scores)

        return max(valid, key=harmonic_mean)

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def compute_agreement_score(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute agreement metrics across multiple solutions.

    Measures consensus in correctness predictions.

    Args:
        solutions: List of solutions with PRM scores

    Returns:
        Dictionary with agreement metrics
    """
    if not solutions:
        return {"error": "No solutions"}

    # Count correct predictions
    correct_count = sum(1 for s in solutions if s.get("score", 0) == 1)
    total_count = len(solutions)

    # Get unique answers
    answers = [s.get("answer") for s in solutions if "answer" in s]
    unique_answers = len(set(answers))

    # Calculate consensus
    consensus_score = correct_count / total_count if total_count > 0 else 0

    return {
        "total_solutions": total_count,
        "correct_solutions": correct_count,
        "consensus_score": consensus_score,
        "unique_answers": unique_answers,
        "high_agreement": consensus_score > 0.7,
    }


def analyze_step_scores(
    plus_probs: List[float]
) -> Dict[str, Any]:
    """
    Analyze PRM scores across steps.

    Computes statistics about score distribution.

    Args:
        plus_probs: List of plus probabilities for each step

    Returns:
        Dictionary with analysis
    """
    if not plus_probs:
        return {"error": "No probabilities"}

    import statistics

    return {
        "num_steps": len(plus_probs),
        "min": min(plus_probs),
        "max": max(plus_probs),
        "mean": statistics.mean(plus_probs),
        "median": statistics.median(plus_probs),
        "stdev": statistics.stdev(plus_probs) if len(plus_probs) > 1 else 0,
        "variance": statistics.variance(plus_probs) if len(plus_probs) > 1 else 0,
        "monotonic_increase": all(
            plus_probs[i] <= plus_probs[i+1] for i in range(len(plus_probs)-1)
        ),
    }
