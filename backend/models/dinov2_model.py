import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from utils import safe_log_event

class DINOv2Preprocessor:
    """
    Offline image embedder using DINOv2 for feature extraction.
    Loads model + processor and extracts CLS token as vector embedding.
    Intended for FAISS indexing or local testing only.
    """
    def __init__(self, 
        model_name: str = "facebook/dinov2-base",
        shape: tuple = (224,224)):
        """
        Initialize processor with a fixed shape and set up device.

        Args:
            model_name (str): HuggingFace model name.
            shape (Tuple[int, int]): Image input size (H, W).
        """

        self.model_name = model_name
        self.shape = shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                size = {"height": shape[0], "width": shape[1]})
            safe_log_event("info", "DINOv2 processor loaded", {"model_name": model_name})
        except Exception as e:
            safe_log_event("error", "Failed to load DINOv2 processor", {"exception": str(e)})
            raise

    def process(self, image: Image.Image) -> dict:
        """
        Prepares an image for Triton model inference.

        Args:
            image (PIL.Image): Input image

        Returns:
            dict: Processed image tensor dict with 'pixel_values'
        """
        image = image.convert("RGB")
        return self.processor(images=image, return_tensors="pt")


class DINOv2EmbeddingGenerator:
    def __init__(self, model_name="facebook/dinov2-base", shape=(224, 224)):
        """
        Loads model and processor on specified device.

        Args:
            model_name (str): HuggingFace model name.
            shape (Tuple[int, int]): Input image size (H, W).
        """
        self.shape = shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, size={"height": shape[0], "width": shape[1]}
        )
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        safe_log_event("info", "DINOv2 model loaded for local embedding", {
            "model_name": model_name,
            "device": str(self.device)
        })

    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Generates a single [1, 768] image embedding using the CLS token.

        Args:
            image (PIL.Image): Input image

        Returns:
            np.ndarray: Image embedding (CLS token), shape [1, 768].
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        expected_shape = (1, 3, self.shape[0], self.shape[1])
        assert pixel_values.shape == expected_shape, f"Expected {expected_shape}, got {pixel_values.shape}" 

        with torch.no_grad():
            outputs = self.model(pixel_values.to(self.device))
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        safe_log_event("info", "Image embedding generated", {"shape": cls_embedding.shape})
        return cls_embedding.cpu().numpy()
