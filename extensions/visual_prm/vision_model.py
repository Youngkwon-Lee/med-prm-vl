"""
Vision Model Integration
========================
InternVL3-8B integration for Visual PRM.

Model Selection Rationale:
  ✓ InternVL3-8B (8B parameters)
  ✓ Validated in VisualPRM paper (arXiv 2503.10291)
  ✓ Matches Med-PRM size (~8B)
  ✓ Supports medical images
  ✓ HuggingFace native support

Alternative Models (future):
  - Llava-1.5 (7B parameters)
  - Qwen-VL (10B parameters)
  - Flamingo (80B - too large)

Implementation Status: 🟡 Framework ready
"""

from typing import Optional, Dict, Any, Tuple, Union
import torch


class VisionPRMModel:
    """
    Vision-aware PRM model using InternVL3.

    Features:
      - Loads text-only or vision model based on configuration
      - Backward compatible with text-only Med-PRM
      - Optional vision encoder freezing for fast training
      - Supports both inference and training modes
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        use_vision: bool = True,
        freeze_vision_encoder: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True
    ):
        """
        Initialize Vision PRM model.

        Args:
            model_name: HuggingFace model identifier
            use_vision: Whether to use vision model (False = text-only)
            freeze_vision_encoder: Freeze vision encoder during training
            dtype: Data type (bfloat16, float16)
            device_map: Device mapping (auto, cuda, cpu)
            trust_remote_code: Allow remote code execution
        """
        self.model_name = model_name
        self.use_vision = use_vision
        self.freeze_vision_encoder = freeze_vision_encoder
        self.dtype = dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.processor = None
        self.tokenizer = None

        if use_vision:
            self._load_vision_model()
        else:
            self._load_text_model()

    def _load_vision_model(self):
        """Load InternVL3 vision-language model."""
        print(f"🖼️  Loading vision model: {self.model_name}")

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        try:
            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            # Get tokenizer
            self.tokenizer = self.processor.tokenizer

            print(f"✅ Vision model loaded: {type(self.model).__name__}")
            print(f"   Parameters: {self.model.num_parameters():,}")

            # Freeze vision encoder if requested
            if self.freeze_vision_encoder:
                self._freeze_vision_encoder()

        except Exception as e:
            raise RuntimeError(f"Failed to load vision model: {e}")

    def _load_text_model(self):
        """Fallback to text-only model (backward compatibility)."""
        print(f"📝 Loading text-only model (vision disabled)")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        try:
            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

            # Use standard text model
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=self.device_map
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct"
            )

            print(f"✅ Text model loaded: {type(self.model).__name__}")

        except Exception as e:
            raise RuntimeError(f"Failed to load text model: {e}")

    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        if not self.use_vision or not hasattr(self.model, 'vision_tower'):
            return

        print("🔒 Freezing vision encoder...")
        for param in self.model.vision_tower.parameters():
            param.requires_grad = False

        print(f"   Vision parameters frozen: {sum(p.numel() for p in self.model.vision_tower.parameters()):,}")

    def score_explanation_with_image(
        self,
        question: str,
        explanation: str,
        image_tensor: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score explanation with optional image context.

        This is the main scoring interface combining text and image.

        Args:
            question: Medical question
            explanation: Proposed explanation
            image_tensor: Optional image tensor [C, H, W]
            system_prompt: System instruction

        Returns:
            Dictionary with scores:
                - final_score: Final step probability
                - min_score: Minimum step probability
                - step_scores: Per-step probabilities
        """
        if not self.use_vision or image_tensor is None:
            # Fallback to text-only scoring
            return self._score_text_only(question, explanation, system_prompt)

        print("🖼️  + 📝 Scoring with image context...")

        # TODO: Implement multimodal scoring
        # 1. Embed image with vision encoder
        # 2. Add image embeddings to text prompt
        # 3. Score as in standard PRM

        return {"error": "Multimodal scoring not yet implemented"}

    def _score_text_only(
        self,
        question: str,
        explanation: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Score using text-only mode."""
        # Placeholder: would call standard PRM scoring
        return {
            "final_score": None,
            "min_score": None,
            "step_scores": [],
            "note": "Text-only scoring not yet implemented"
        }

    def forward(self, *args, **kwargs):
        """Forward pass (delegates to underlying model)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model(*args, **kwargs)

    def to(self, device: str) -> "VisionPRMModel":
        """Move model to device."""
        if self.model:
            self.model.to(device)
        return self

    def eval(self) -> "VisionPRMModel":
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self

    def train(self) -> "VisionPRMModel":
        """Set model to training mode."""
        if self.model:
            self.model.train()
        return self


class VisualPRMTrainer:
    """
    Trainer for multimodal PRM.

    Extends standard PRM training with image input support.

    Status: 🔴 Not implemented
    """

    def __init__(
        self,
        model: VisionPRMModel,
        train_dataset,
        eval_dataset=None,
        learning_rate: float = 1e-5,
        freeze_vision: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: VisionPRMModel instance
            train_dataset: Training dataset with images
            eval_dataset: Evaluation dataset
            learning_rate: Learning rate for LoRA layers
            freeze_vision: Whether to freeze vision encoder
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.freeze_vision = freeze_vision

    def train(self, num_epochs: int = 3, batch_size: int = 4) -> Dict[str, Any]:
        """
        Train model on multimodal dataset.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training results
        """
        # TODO: Implement training loop
        return {"error": "Training not yet implemented"}


# Utility functions for inference

def load_vision_model_for_inference(
    model_name: str = "OpenGVLab/InternVL3-8B",
    dtype: str = "bfloat16"
) -> VisionPRMModel:
    """
    Convenience function to load model for inference.

    Args:
        model_name: Model identifier
        dtype: Data type

    Returns:
        VisionPRMModel in eval mode
    """
    model = VisionPRMModel(
        model_name=model_name,
        use_vision=True,
        freeze_vision_encoder=False,
        dtype=dtype
    )
    model.eval()
    return model


def compare_text_vs_vision_scoring(
    question: str,
    explanation: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare text-only vs multimodal scoring on same input.

    Useful for evaluation and understanding image contribution.

    Args:
        question: Medical question
        explanation: Proposed explanation
        image_path: Optional path to medical image

    Returns:
        Comparison results
    """
    # Load both models
    text_model = VisionPRMModel(use_vision=False)
    vision_model = VisionPRMModel(use_vision=True)

    # Score text-only
    text_scores = text_model.score_explanation_with_image(question, explanation)

    # Score with vision (if image provided)
    if image_path:
        from .image_processor import MedicalImageProcessor
        processor = MedicalImageProcessor()
        image_tensor = processor.preprocess(image_path)
        vision_scores = vision_model.score_explanation_with_image(
            question, explanation, image_tensor
        )
    else:
        vision_scores = {"error": "No image provided"}

    return {
        "text_scores": text_scores,
        "vision_scores": vision_scores,
        "improvement": "To be analyzed"
    }
