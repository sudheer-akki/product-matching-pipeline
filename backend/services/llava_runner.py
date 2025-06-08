from .triton_client import infer_llava
from core import log_event

async def generate_caption(
    image_bytes: bytes,
    prompt: str = "Describe this image.",
    max_tokens: int = 128,
    temperature: float = 0.2,
    top_k: int = 1,
    freq_penalty: float = 0.0,
    seed: int = 42
) -> str:
    """
    Calls the LLaVA inference service (via Triton) to generate a caption from an image.
    """
    try:
        log_event("info", "Generating caption via LLaVA", {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "freq_penalty": freq_penalty,
            "seed": seed
        })

        caption = infer_llava(
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            freq_penalty=freq_penalty,
            seed=seed
        )

        log_event("info", "Caption generated successfully", {"caption": caption})
        return caption

    except Exception as e:
        log_event("error", "LLaVA caption inference failed", {"exception": str(e)})
        return "Caption generation failed."
