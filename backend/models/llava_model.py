import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from typing import List, Union


class LLaVAProcessor:
    """
    Prepares image and prompt inputs for LLaVA (e.g., llava-1.5-7b-hf) models.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        try:
            self.model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaVA processor: {e}")

    def prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Formats the prompt + image into model-ready NumPy arrays.

        Returns:
            input_ids (np.ndarray): [1, seq_len], dtype=int64
            pixel_values (np.ndarray): [1, 3, 224, 224], dtype=float32
        """
        try:
            # LLaVA expects this structured input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            formatted_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].numpy().astype("int64")
            pixel_values = inputs["pixel_values"].numpy().astype("float32")

            return input_ids, pixel_values

        except Exception as e:
            raise RuntimeError(f"LLaVA input preparation failed: {e}")
        

class LLaVACaptioner:
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        #     low_cpu_mem_usage=True
        # ).to(self.device)

    def tokenize_prompt(self, prompt: str) -> List[int]:
        """
        Tokenizes a given text prompt using the processor's tokenizer.
        """
        inputs = self.processor.tokenizer(prompt, return_tensors="pt", truncation=True)
        return inputs["input_ids"].squeeze().tolist()

    def generate_caption(self, image: Union[str, Image.Image], prompt: str = "Describe this image.") -> str:
        """
        Generates a caption for the given image using the provided or default prompt.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=chat_prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device.type == "cuda" else torch.float32)

        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        return self.processor.decode(output[0][2:], skip_special_tokens=True)
