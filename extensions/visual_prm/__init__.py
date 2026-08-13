"""
Visual PRM Extension
====================
Extends Med-PRM to support medical images (X-ray, CT, MRI) via InternVL3.

Modules:
  - image_processor: DICOM and image file preprocessing
  - vision_model: InternVL3 model integration
  - multimodal_prm: Vision-aware PRM scoring

Usage:
    from extensions.visual_prm import VisionPRMModel, MedicalImageProcessor

    # Load vision model
    model = VisionPRMModel(
        model_name="OpenGVLab/InternVL3-8B",
        freeze_encoder=False
    )

    # Process medical image
    processor = MedicalImageProcessor()
    image_tensor = processor.preprocess_dicom("chest_xray.dcm")

    # Score with image context
    scores = model.score_explanation_with_image(
        question="What is the diagnosis?",
        explanation="The patient has...",
        image_path="chest_xray.dcm"
    )

Features:
  ✓ DICOM support (CT, MRI, X-ray)
  ✓ Standard formats (PNG, JPG, JPEG)
  ✓ Automatic normalization
  ✓ Batch processing ready
  ✓ InternVL3-8B as backbone (from VisualPRM paper)
  ✓ Medical-specific preprocessing

Timeline:
  Phase 1 (2 weeks): Foundation setup + MIMIC-CXR integration
  Phase 2 (2 weeks): Training + evaluation
  Phase 3 (4 weeks): Optimization + 3D imaging (CT/MRI)
  Phase 4 (8+ weeks): Production deployment

Status: 🟡 Planned (Infrastructure ready, implementation pending)
"""

__version__ = "0.1.0"
__all__ = ["VisionPRMModel", "MedicalImageProcessor"]

# These will be available once fully implemented
# from .vision_model import VisionPRMModel
# from .image_processor import MedicalImageProcessor
# from .multimodal_prm import MultimodalPRM
