import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort


class DINOv2ONNXWrapper(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token [B, 768]
        return cls_embedding
    
def export_dinov2_to_onnx(model_name="facebook/dinov2-base", onnx_path="dinov2_cls.onnx"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = DINOv2ONNXWrapper(model_name)
    model.eval()
    # Dummy input
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    inputs = processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # Shape: (1, 3, 224, 224)

    # Export
    torch.onnx.export(
        model,
        (pixel_values,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["cls_embedding"], # [B, 768]
        dynamic_axes={
            "pixel_values": {0: "batch_size"}, 
            "cls_embedding": {0: "batch_size"}
        },
        opset_version=14
    )


def infer_dinov2_onnx(onnx_path, image: Image.Image, model_name="facebook/dinov2-base"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = processor(images=image.convert("RGB"), return_tensors="np")
    pixel_values = inputs["pixel_values"]

    session = ort.InferenceSession(onnx_path)
    outputs = session.run(None, {"pixel_values": pixel_values})
    return outputs[0]  # shape: (1, 768)

if __name__ == "__main__":

    export_dinov2_to_onnx("facebook/dinov2-base", "dinov2.onnx")

    #image = Image.open("scripts/test.jpg")

    #embedding = infer_dinov2_onnx("models/engines/dinov2/dinov2.onnx", image)
    #print("DINOv2 embedding shape:", embedding.shape)
