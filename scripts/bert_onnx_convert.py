import sys, os
import onnxruntime as ort
from transformers import BertTokenizer, BertModel
import torch
from torch import nn

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models import BertWithMeanPooling
    

def export_onnx_model(model, tokenizer, onnx_path="bert_meanpooled.onnx"):
    dummy_input = tokenizer("This is a sample sentence.", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["pooled_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "pooled_output": {0: "batch_size"}
        },
        opset_version=14
    )

def run_onnx_inference(onnx_path, tokenizer, text):
    # Prepare input
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Load model
    session = ort.InferenceSession(onnx_path)

    # Run
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    embedding = outputs[0]  # shape: (1, 768)
    return embedding


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertWithMeanPooling()
    model.eval()

    ONNX_PATH = "models/engines/bert/bert.onnx"
    export_onnx_model(model, tokenizer, ONNX_PATH)

    # Test inference
    embedding = run_onnx_inference(ONNX_PATH, tokenizer, "This is a sample sentence.")
    print("ONNX output shape:", embedding.shape)
    print("Mean pooled embedding (first 5 dims):", embedding[0][:5])
