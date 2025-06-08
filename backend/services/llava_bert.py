from fastapi import Request
from core import log_event
import numpy as np

async def llava_bert_pipeline(
    request: Request,
    image_bytes: bytes,
    prompt: str = "Describe this image.",
    max_tokens: int = 128,
    temperature: float = 0.2,
    top_k: int = 1,
    freq_penalty: float = 0.0,
    seed: int = 42
) -> tuple[str, np.ndarray]:
    """
    Calls the LLaVA inference service (via Triton) to generate a caption and BERT embedding from an image.

    Uses batching via request.app.state.batcher_llava_bert
    """
    try:
        log_event("info", "Generating Embeddings via LLaVA -> Bert", {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "freq_penalty": freq_penalty,
            "seed": seed
        })

        input_payload = {
            "image_bytes": image_bytes,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "freq_penalty": freq_penalty,
            "seed": seed
        }

        # Sending Payload to Llava -> Bert on Triton Server
        caption, embedding = await request.app.state.batcher_llava_bert.submit(input_payload)
        log_event("info", "Llava-Bert Embeddings pipeline completed", 
                  {"Embeddings": embedding.shape, "caption": caption})
        return caption, embedding

    except Exception as e:
        log_event("error", "LLaVA-Bert inference failed", {"exception": str(e)})
        return "Caption generation failed."
