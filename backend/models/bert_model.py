import torch
import numpy as np
from typing import Union, List
from transformers import BertTokenizer, BertModel
from torch import nn
from utils import safe_log_event

class BertWithMeanPooling(nn.Module):
    """
    A wrapper around Hugging Face's BERT that adds
    mean pooling with attention masking on the last hidden layer.
    Used to export BERT to ONNX with embedded postprocessing.
    """

    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        """
        Perform forward pass and apply attention-masked mean pooling.

        Args:
            input_ids (Tensor): [B, T] token IDs
            attention_mask (Tensor): [B, T] mask for valid tokens

        Returns:
            Tensor: [B, 768] mean pooled sentence embeddings
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, 768]

        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_hidden = (last_hidden * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9) # Avoiding Zero Division
        mean_pooled = sum_hidden / sum_mask
        return mean_pooled  # [B, 768]


class BERTEmbeddingGenerator:
    """
    Used to generate BERT embeddings offline for tasks like
    FAISS indexing. Includes mean pooling and fixed padding.
    """
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + tokenizer
        self.model = BertWithMeanPooling(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> np.ndarray:
        """
        Converts one or more texts into BERT mean pooled embeddings.

        Args:
            texts (str or List[str]): Input sentence(s)

        Returns:
            np.ndarray: Embeddings of shape [B, 768]
        """
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = self.model(**inputs)  # [1, 768]
        return embedding.cpu().numpy()


class BERTTextProcessor:
    """
    Lightweight tokenizer-only class used at FastAPI runtime.
    It tokenizes text for Triton inference and avoids loading the full BERT model.
    """
    def __init__(self, 
            model_name: str = "bert-base-uncased",
            max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.max_length <= self.tokenizer.model_max_length, \
        f"max_length {self.max_length} exceeds model's limit {self.tokenizer.model_max_length}"

        safe_log_event("info", "BERT tokenizer initialized", {"model_name": model_name})

    def process(self, texts: Union[str, List[str]]) -> dict:
        """
        Tokenizes input text(s) with fixed-length padding for BERT-compatible models.

        Args:
            texts (str or List[str]): Input text or list of texts

        Returns:
            dict: Tokenized tensors with keys: input_ids, attention_mask
        """
        return self.tokenizer(
            texts,
            padding= "max_length", #Fixed Length Padding
            max_length= self.max_length,
            truncation=True,
            return_tensors="pt"
        )