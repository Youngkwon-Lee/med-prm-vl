"""
Model utilities for Med-PRM
============================
Handles model loading, tokenizer setup, and token ID extraction.

Centralizes model initialization logic used across:
  - python/2_training.py (lines 91-113)
  - python/4_scoring_PRM.py (lines 132-157)
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .constants import (
    DEFAULT_DTYPE,
    DEFAULT_ATTENTION_IMPL,
    MODEL_CONFIGS,
)


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = DEFAULT_DTYPE,
    attention_impl: Optional[str] = DEFAULT_ATTENTION_IMPL,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = True
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from HuggingFace.

    Handles:
      - Device mapping (auto, single GPU, multi-GPU)
      - Data type conversion (bfloat16, float16)
      - Attention implementation (flash_attention_2, eager)
      - Pad token setup (handles models without pad token)
      - Gradient checkpointing for memory efficiency

    Args:
        model_path: Model path or identifier
        dtype: Data type ("bfloat16" or "float16")
        attention_impl: Attention implementation (None to disable)
        device_map: Device mapping strategy
        hf_token: HuggingFace API token (for gated models)
        gradient_checkpointing: Enable gradient checkpointing for training
        trust_remote_code: Allow remote code execution (for some models)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If dtype is not supported
        RuntimeError: If model loading fails

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     dtype="bfloat16"
        ... )
        >>> model.eval()
    """
    if hf_token:
        login(hf_token)

    print(f"🔄 Loading model: {model_path}")
    print(f"   dtype: {dtype}")
    print(f"   attention: {attention_impl or 'eager'}")

    # Convert dtype string to torch dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'bfloat16' or 'float16'")

    # Prepare kwargs for from_pretrained
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    # Add attention implementation if specified
    if attention_impl:
        model_kwargs["attn_implementation"] = attention_impl

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_path}: {e}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Enable gradient checkpointing for training
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Set pad token (required for models like LLaMA that don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ⚠️  Set pad_token to eos_token")

    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Parameters: {model.num_parameters():,}")
    print(f"   Tokenizer: {type(tokenizer).__name__}")

    return model, tokenizer


def get_token_ids(
    tokenizer: Any,
    tokens: List[str]
) -> Dict[str, int]:
    """
    Extract token IDs for special tokens.

    Safely gets token IDs for special tokens used in PRM scoring.

    Args:
        tokenizer: HuggingFace tokenizer
        tokens: List of token strings (e.g., [" +", " -", " ки"])

    Returns:
        Dictionary mapping token string to token ID

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> ids = get_token_ids(tokenizer, [" +", " -"])
        >>> ids[" +"] > 0
        True
    """
    token_ids = {}

    for token in tokens:
        try:
            encoded = tokenizer(token, add_special_tokens=False)
            token_id = encoded["input_ids"][0] if encoded["input_ids"] else None

            if token_id is not None:
                token_ids[token] = token_id
            else:
                print(f"⚠️  Warning: Could not get token ID for '{token}'")
        except Exception as e:
            print(f"⚠️  Error getting token ID for '{token}': {e}")

    return token_ids


def get_plus_minus_ids(tokenizer: Any) -> Tuple[int, int]:
    """
    Get token IDs for '+' and '-' markers used in PRM scoring.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (plus_id, minus_id)

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> plus_id, minus_id = get_plus_minus_ids(tokenizer)
        >>> print(f"Plus ID: {plus_id}, Minus ID: {minus_id}")
    """
    ids = get_token_ids(tokenizer, [" +", " -"])

    plus_id = ids.get(" +")
    minus_id = ids.get(" -")

    if plus_id is None or minus_id is None:
        raise ValueError("Could not extract '+' and '-' token IDs from tokenizer")

    print(f"✅ Token IDs extracted:")
    print(f"   plus_id ({tokenizer.convert_ids_to_tokens([plus_id])[0]}): {plus_id}")
    print(f"   minus_id ({tokenizer.convert_ids_to_tokens([minus_id])[0]}): {minus_id}")

    return plus_id, minus_id


def setup_inference_mode(model: Any, use_half_precision: bool = True) -> Any:
    """
    Setup model for inference (eval mode, no gradients).

    Args:
        model: PyTorch model
        use_half_precision: Whether model is in half precision

    Returns:
        Model in inference mode
    """
    model.eval()

    if use_half_precision:
        # Models in half precision can benefit from disabled cuDNN benchmarking
        torch.backends.cudnn.benchmark = False

    return model


def estimate_model_memory(
    model: Any,
    batch_size: int = 1,
    seq_length: int = 4096,
    dtype: str = "bfloat16"
) -> Dict[str, float]:
    """
    Estimate memory usage for model inference.

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_length: Sequence length
        dtype: Data type

    Returns:
        Dictionary with memory estimates in GB

    Examples:
        >>> model, _ = load_model_and_tokenizer("meta-llama/Llama-3.1-8B")
        >>> memory = estimate_model_memory(model, batch_size=8)
        >>> print(f"Estimated GPU memory: {memory['total']:.1f} GB")
    """
    bytes_per_param = 2 if dtype == "bfloat16" else 2  # Both are 2 bytes

    num_params = model.num_parameters()
    model_size = num_params * bytes_per_param / (1024 ** 3)  # GB

    # Activation memory during forward pass
    activation_size = batch_size * seq_length * model.config.hidden_size * bytes_per_param / (1024 ** 3)

    # Gradient memory during backward pass
    gradient_size = model_size  # Same as model size

    return {
        "model": model_size,
        "activation_forward": activation_size,
        "gradient_backward": gradient_size,
        "total_training": model_size + activation_size + gradient_size,
        "total_inference": model_size + activation_size,
    }
