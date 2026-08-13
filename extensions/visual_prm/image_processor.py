"""
Medical Image Preprocessing
============================
Handles preprocessing of DICOM and standard image formats for Visual PRM.

Supports:
  - DICOM files (CT, X-ray, MRI)
  - PNG, JPG, JPEG standard formats
  - Automatic window-level adjustment (DICOM-specific)
  - Normalization and resizing

Implementation Status: 🟡 Framework ready (actual DICOM support needs pydicom)
"""

from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np


class MedicalImageProcessor:
    """
    Preprocesses medical images for vision models.

    Handles different modalities:
      - DICOM (CT, X-ray, MRI)
      - Standard image formats (PNG, JPG)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        normalize_method: str = "imagenet"
    ):
        """
        Initialize image processor.

        Args:
            target_size: Target image resolution
            normalize: Whether to normalize pixel values
            normalize_method: Normalization approach
                - "imagenet": ImageNet statistics
                - "minmax": Min-max scaling to [0,1]
                - "dicom": DICOM-specific normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        self.normalize_method = normalize_method

    def preprocess(self, image_path: str) -> Any:
        """
        Preprocess image file.

        Auto-detects format (DICOM vs standard) and processes accordingly.

        Args:
            image_path: Path to image file

        Returns:
            torch.Tensor ready for vision model

        Examples:
            >>> processor = MedicalImageProcessor()
            >>> tensor = processor.preprocess("chest_xray.dcm")
            >>> tensor.shape
            torch.Size([3, 224, 224])
        """
        image_path = Path(image_path)

        if image_path.suffix.lower() == ".dcm":
            return self.preprocess_dicom(image_path)
        else:
            return self.preprocess_standard(image_path)

    def preprocess_dicom(self, dicom_path: Union[str, Path]) -> Any:
        """
        Preprocess DICOM file.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            torch.Tensor in shape [3, H, W] (RGB for compatibility)

        Note:
            Requires pydicom: pip install pydicom
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom required for DICOM support: pip install pydicom")

        try:
            import torch
        except ImportError:
            raise ImportError("torch required: pip install torch")

        # Load DICOM
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array

        # Convert to uint8 for standard image processing
        if pixel_array.dtype == np.int16:
            # Handle signed pixel data (common in X-ray)
            pixel_array = np.clip(pixel_array, -1024, 3071)  # Hounsfield units
            pixel_array = ((pixel_array + 1024) / 4096 * 255).astype(np.uint8)
        elif pixel_array.max() > 255:
            pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)

        # Convert grayscale to RGB (repeat channels)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)

        # Resize
        from PIL import Image
        img = Image.fromarray(pixel_array)
        img = img.resize(self.target_size, Image.BILINEAR)
        img_array = np.array(img).astype(np.float32)

        # Normalize
        if self.normalize:
            img_array = self._normalize(img_array)

        # Convert to torch tensor and rearrange to [C, H, W]
        tensor = torch.from_numpy(img_array)
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        return tensor

    def preprocess_standard(self, image_path: Union[str, Path]) -> Any:
        """
        Preprocess standard image format (PNG, JPG).

        Args:
            image_path: Path to image file

        Returns:
            torch.Tensor in shape [3, H, W]
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pillow required: pip install pillow")

        try:
            import torch
        except ImportError:
            raise ImportError("torch required: pip install torch")

        # Open and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # Resize
        img = img.resize(self.target_size, Image.BILINEAR)

        # Convert to array
        img_array = np.array(img).astype(np.float32)

        # Normalize
        if self.normalize:
            img_array = self._normalize(img_array)

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(img_array)
        tensor = tensor.permute(2, 0, 1)

        return tensor

    def _normalize(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize image array.

        Args:
            img_array: Image as numpy array

        Returns:
            Normalized image
        """
        if self.normalize_method == "imagenet":
            # ImageNet statistics (from torchvision)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = img_array / 255.0
            img_array = (img_array - mean) / std
            return img_array

        elif self.normalize_method == "minmax":
            # Min-max scaling to [0, 1]
            return img_array / 255.0

        elif self.normalize_method == "dicom":
            # DICOM-specific: already handled in preprocess_dicom
            return img_array / 255.0

        return img_array

    @staticmethod
    def batch_preprocess(image_paths: List[str]) -> Any:
        """
        Preprocess multiple images for batch processing.

        Args:
            image_paths: List of image file paths

        Returns:
            Batched torch.Tensor of shape [B, 3, H, W]
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required: pip install torch")

        processor = MedicalImageProcessor()
        tensors = [processor.preprocess(path) for path in image_paths]
        return torch.stack(tensors, dim=0)


class WindowLevelAdjustment:
    """
    Apply window-level adjustment for DICOM images.

    Common medical window settings:
      - Abdomen: center=40, width=350
      - Chest: center=50, width=400
      - Bone: center=450, width=1500
      - Brain: center=40, width=100
    """

    @staticmethod
    def adjust(
        pixel_array: np.ndarray,
        window_center: int,
        window_width: int
    ) -> np.ndarray:
        """
        Apply window-level adjustment.

        Args:
            pixel_array: Pixel data (Hounsfield units)
            window_center: Center of window
            window_width: Width of window

        Returns:
            Adjusted pixel array [0, 255]
        """
        lower = window_center - window_width // 2
        upper = window_center + window_width // 2

        adjusted = np.clip(pixel_array, lower, upper)
        adjusted = ((adjusted - lower) / (upper - lower) * 255).astype(np.uint8)

        return adjusted

    @staticmethod
    def auto_detect_window(modality: str) -> Tuple[int, int]:
        """
        Auto-detect window settings by modality.

        Args:
            modality: DICOM modality (CT, XC, MR, US, etc.)

        Returns:
            Tuple of (center, width)
        """
        windows = {
            "CT": (40, 400),  # Chest/Abdomen default
            "XC": (50, 400),  # X-ray
            "MR": (128, 256),  # MRI (generic)
            "US": (128, 256),  # Ultrasound
        }
        return windows.get(modality[:2].upper(), (50, 400))
