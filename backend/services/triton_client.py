import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io
from typing import Tuple, List
from tritonclient.utils import np_to_triton_dtype
from core import log_event, load_config
import traceback

config = load_config()
TRITON_URL = config.api.triton_url #"triton:8000" 


def is_triton_available(url):
    try:
        client = httpclient.InferenceServerClient(url=url, ssl=False)
        client.is_server_live()
        client.server_url = url
        return client
    except Exception:
        return None

client = is_triton_available("localhost:8000")

if client is None:
    client = is_triton_available(TRITON_URL)
    if client is None:
        raise RuntimeError(f"Could not connect to Triton at localhost:8000 or {TRITON_URL}")

log_event("info", f"Connected to Triton at: {client.server_url}")


USE_DUMMY_BERT = False  # Set this to False when real Triton is available
USE_DUMMY_LLAVA = False  # Toggle for test mode
USE_DUMMY_DINOV2 = False  # Toggle this flag based on environment or config


# ------------------------- BERT -------------------------

async def infer_bert(input_text: dict) -> dict:
    """
    Sends a batch of inputs to the Triton server for BERT embeddings.
    input_ids_list: list of np.array, each shape [seq_len]
    Returns: np.array of shape [B, 768]
    """
    if isinstance(input_text, list):
        input_text = input_text[0]

    input_ids = input_text["input_ids"]
    attention_mask = input_text["attention_mask"]

    if USE_DUMMY_BERT:
        log_event("info", "Returning dummy BERT embedding (Triton not active)")
        # Return a dummy embedding (e.g., 768-dim vector per input)
        batch_size = input_ids.shape[0]
        dummy_embedding = np.random.rand(batch_size, 768).astype(np.float32) #[1, 768]
        return dummy_embedding

    try:
        input_ids = input_ids.astype(np.int64)  # [B, T]
        attention_mask = attention_mask.astype(np.int64) # [B, T]
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        outputs = [httpclient.InferRequestedOutput("pooled_output")]

        response = client.infer(model_name="bert", inputs=inputs, outputs=outputs)
        embedding  = response.as_numpy("pooled_output") # shape: [B, S, 768]
        #embedding = embedding.astype("float32").reshape(1, -1)
        log_event("info", "BERT Triton inference successful", {"embedding_shape": embedding.shape})
        return {
            "embedding": embedding,
            "embedding_shape": list(embedding.shape)
        }
    except Exception as e:
        log_event("error", "BERT Triton inference failed", {"exception": str(e)})
        raise e


# ------------------------- DINOv2 -------------------------

async def infer_dinov2(input_data: dict ) -> dict:
    assert isinstance(input_data, dict)
    assert "pixel_values" in input_data
    try:
        pixel_values = input_data["pixel_values"]
        pixel_values = np.asarray(pixel_values) 
        batch_size = pixel_values.shape[0]
        
        if USE_DUMMY_DINOV2:
            log_event("info", "Returning dummy DINOv2 embedding (Triton not active)")
            dummy_embedding = np.random.rand(batch_size, 768).astype(np.float32)  # [B, 768]
            return dummy_embedding
        
        inputs = [
            httpclient.InferInput("pixel_values", pixel_values.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(pixel_values)
        outputs = [httpclient.InferRequestedOutput("cls_embedding")]
        response = client.infer(model_name="dinov2", inputs=inputs, outputs=outputs)
        embedding = response.as_numpy("cls_embedding") #[B, 768]
        log_event("info", "DINOv2 Triton inference successful", {"embedding_shape": embedding.shape})
        return  {
            "embedding": embedding,
            "embedding_shape": list(embedding.shape)
        }
    except Exception as e:
        log_event("error", "DINOv2 Triton inference failed", {"exception": str(e)})
        raise e

# ------------------------- LLaVA -------------------------

def prepare_tensor(name, arr, dtype):
    infer_input = httpclient.InferInput(name, arr.shape, np_to_triton_dtype(dtype))
    infer_input.set_data_from_numpy(arr)
    return infer_input

def prepare_image_tensor(image_bytes: bytes) -> np.ndarray:
    """
    Prepare image tensor for Triton Python backend.
    Args:
        image_bytes: Raw bytes of the image (e.g., from open("img.jpg", "rb").read())
    Returns:
        np.ndarray of shape (1,) and dtype object, as expected by Triton TYPE_STRING
    """
    return np.array([image_bytes], dtype=object)


def build_input_tensors(image_bytes, prompt, max_tokens=128, temperature=0.2, top_k=1, freq_penalty=0.0, seed=42):
    return [
        prepare_tensor("image", prepare_image_tensor(image_bytes), np.object_),
        prepare_tensor("prompt", np.array([prompt.encode("utf-8")], dtype=np.object_), np.object_),
        prepare_tensor("max_tokens", np.array([max_tokens], dtype=np.int32), np.int32),
        prepare_tensor("temperature", np.array([temperature], dtype=np.float32), np.float32),
        prepare_tensor("top_k", np.array([top_k], dtype=np.int32), np.int32),
        prepare_tensor("frequency_penalty", np.array([freq_penalty], dtype=np.float32), np.float32),
        prepare_tensor("seed", np.array([seed], dtype=np.uint64), np.uint64),
    ]

def resize_image_bytes(image_bytes, size=(336, 336)):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize(size)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    return buf.getvalue()


def infer_llava_bert(input_dict: dict) -> Tuple[str, np.ndarray]:
    """
    Inference for LLaVA-BERT pipeline using Triton, for a single input.

    Args:
        input_dict (dict): Must contain keys:
            - image_bytes, prompt, max_tokens, temperature, top_k, freq_penalty, seed

    Returns:
        Tuple[str, np.ndarray]: (caption, embedding) for the input
    """
    try:
        log_event("info", "infer_llava_bert: Starting inference", {
            "input_keys": list(input_dict.keys()),
            "image_bytes_size": len(input_dict["image_bytes"]) if "image_bytes" in input_dict else None,
            "prompt": input_dict.get("prompt", "")
        })

        if USE_DUMMY_LLAVA:
            dummy_embedding = np.random.rand(768).astype(np.float32)
            dummy_caption = "The person wears a short-sleeve T-shirt with solid color patterns and long pants."
            log_event("info", "Returning dummy LLaVA caption (Triton not active)", {"caption": dummy_caption})
            return dummy_caption, dummy_embedding.copy()

        log_event("info", "LLaVA-BERT single inference started", {})

        resized_bytes = resize_image_bytes(input_dict["image_bytes"], size=(384, 384))
        log_event("info", "Image resized locally", {"resized_bytes_size": len(resized_bytes)})

        prompt_bytes = input_dict["prompt"].encode("utf-8")
        max_tokens = np.array(input_dict.get("max_tokens", 128), dtype=np.int32)
        temperature = np.array(input_dict.get("temperature", 0.2), dtype=np.float32)
        top_k = np.array(input_dict.get("top_k", 1), dtype=np.int32)
        freq_penalty = np.array(input_dict.get("freq_penalty", 0.0), dtype=np.float32)
        seed = np.array(input_dict.get("seed", 42), dtype=np.uint64)

        log_event("info", "Prepared Triton inputs", {
            "prompt_bytes": prompt_bytes,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "freq_penalty": float(freq_penalty),
            "seed": int(seed)
        })

        inputs = [
            prepare_tensor("image", np.array([[resized_bytes]], dtype=np.object_), np.object_),
            prepare_tensor("prompt", np.array([[prompt_bytes]], dtype=np.object_), np.object_),
            prepare_tensor("max_tokens", np.array([[max_tokens]], dtype=np.int32), np.int32)
        ]

        outputs = [
            httpclient.InferRequestedOutput("pooled_output"),
            httpclient.InferRequestedOutput("text")
        ]

        log_event("info", "Sending inference to Triton", {"model_name": "llava_to_bert"})

        # Run inference
        response = client.infer(model_name="llava_to_bert", inputs=inputs, outputs=outputs)

        # Parse outputs (always return the first item in output)
        embedding = response.as_numpy("pooled_output")[0]  # shape: [768]
        caption = response.as_numpy("text")[0].decode("utf-8")

        log_event("info", "LLaVA-BERT inference completed", {
            "caption": caption,
            "embedding_shape": embedding.shape
        })
        return caption, embedding

    except Exception as e:
        log_event("error", "LLaVA-BERT single failed", {
            "exception": str(e),
            "traceback": traceback.format_exc()
        })
        return "Caption generation failed.", np.zeros((768,), dtype=np.float32)

















def infer_llava_bert_batch(batch_inputs: List[dict]) -> List[Tuple[str, np.ndarray]]:
    """
    Batched inference for LLaVA-BERT pipeline using Triton.

    Args:
        batch_inputs (List[dict]): Each dict contains keys:
            - image_bytes, prompt, max_tokens, temperature, top_k, freq_penalty, seed

    Returns:
        List[Tuple[str, np.ndarray]]: (caption, embedding) for each input
    """
    try:
        batch_size = len(batch_inputs)

        if USE_DUMMY_LLAVA:
            dummy_embedding = np.random.rand(batch_size, 768).astype(np.float32) #[1, 768]
            dummy_caption = "The person wears a short-sleeve T-shirt with solid color patterns and long pants."
            log_event("info", "Returning dummy LLaVA caption (Triton not active)", {"caption": dummy_caption})
            return [(dummy_caption, dummy_embedding.copy()) for _ in range(batch_size)]


        log_event("info", "LLaVA-BERT batched inference started", {"batch_size": batch_size})

        # Batch each input field
        image_arr = np.array([prepare_image_tensor(x["image_bytes"])[0] for x in batch_inputs], dtype=np.object_)
        prompt_arr = np.array([x["prompt"].encode("utf-8") for x in batch_inputs], dtype=np.object_)
        max_tokens_arr = np.array([x.get("max_tokens", 128) for x in batch_inputs], dtype=np.int32)
        temperature_arr = np.array([x.get("temperature", 0.2) for x in batch_inputs], dtype=np.float32)
        top_k_arr = np.array([x.get("top_k", 1) for x in batch_inputs], dtype=np.int32)
        freq_penalty_arr = np.array([x.get("freq_penalty", 0.0) for x in batch_inputs], dtype=np.float32)
        seed_arr = np.array([x.get("seed", 42) for x in batch_inputs], dtype=np.uint64)

        # Create Triton inputs using shared functions
        inputs = [
            prepare_tensor("image", image_arr, np.object_),
            prepare_tensor("prompt", prompt_arr, np.object_),
            prepare_tensor("max_tokens", max_tokens_arr, np.int32),
            prepare_tensor("temperature", temperature_arr, np.float32),
            prepare_tensor("top_k", top_k_arr, np.int32),
            prepare_tensor("frequency_penalty", freq_penalty_arr, np.float32),
            prepare_tensor("seed", seed_arr, np.uint64)
        ]

        outputs = [
            httpclient.InferRequestedOutput("pooled_output"),
            httpclient.InferRequestedOutput("text")
        ]

        # Run inference
        response = client.infer(model_name="llava_to_bert", inputs=inputs, outputs=outputs)

        # Parse outputs
        embeddings = response.as_numpy("pooled_output")  # shape: [B, 768]
        captions = response.as_numpy("text")              # shape: [B]

        results = [(captions[i].decode("utf-8"), embeddings[i]) for i in range(batch_size)]
        log_event("info", "LLaVA-BERT batch completed", {"results": len(results)})

        return results

    except Exception as e:
        log_event("error", "LLaVA-BERT batch failed", {"exception": str(e)})
        fallback = [("Caption generation failed.", np.zeros((768,), dtype=np.float32)) for _ in batch_inputs]
        return fallback


def infer_llava_bert_(image_bytes: bytes, 
                prompt: str, 
                max_tokens: int = 128,
                temperature: float = 0.2,
                top_k: int = 1,
                freq_penalty: float = 0.0,
                seed: int = 42) -> Tuple[str, np.ndarray]:
    try:
        inputs = build_input_tensors(
            image_bytes=image_bytes, 
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            freq_penalty=freq_penalty,
            seed=seed)

        outputs = [
            httpclient.InferRequestedOutput("pooled_output"),
            httpclient.InferRequestedOutput("text")
        ]

        response = client.infer(model_name="llava_to_bert", inputs=inputs, outputs=outputs, client_timeout=15)
        embedding  = response.as_numpy("pooled_output") # shape: [B, S, 768]
        caption = response.as_numpy("text") # Llava Caption

        log_event("info", "LLaVA-Berrt Pipeline inference successful", 
                  {"embedding": embedding.shape},
                  {"Caption:", caption})
        return caption, embedding
    
    except Exception as e:
        log_event("error", "LLaVA-Bert Pipeline inference failed", {"exception": str(e)})
        return "LLaVA-Bert Pipeline inference failed."

def infer_llava(image_bytes: bytes, 
                prompt: str, 
                max_tokens: int = 128,
                temperature: float = 0.2,
                top_k: int = 1,
                freq_penalty: float = 0.0,
                seed: int = 42) -> str:
    if USE_DUMMY_LLAVA:
        dummy_caption = "The person wears a short-sleeve T-shirt with solid color patterns and long pants."
        log_event("info", "Returning dummy LLaVA caption (Triton not active)", {"caption": dummy_caption})
        return dummy_caption

    try:
        inputs = build_input_tensors(image_bytes=image_bytes, 
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                freq_penalty=freq_penalty,
                                seed=seed)
        
        outputs = [httpclient.InferRequestedOutput("text")]
        response = client.infer(model_name="llava", inputs=inputs, outputs=outputs, client_timeout=15)
        result = response.as_numpy("text")
        caption = result[0].decode("utf-8")
        log_event("info", "LLaVA inference successful", {"caption": caption})
        return caption
    
    except Exception as e:
        log_event("error", "LLaVA inference failed", {"exception": str(e)})
        return "Caption generation failed."
    
