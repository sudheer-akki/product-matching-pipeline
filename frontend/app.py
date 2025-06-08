import os
import base64
from PIL import Image
from datetime import datetime
import io
import gradio as gr
import requests
from omegaconf import OmegaConf

config_file: str = "../configs/app_config.yaml"
base_dir = os.path.dirname(__file__)
config_path = os.path.abspath(os.path.join(base_dir, config_file))
config = OmegaConf.load(config_path)
print("Frontend Config Loaded")
API_URL = config.api.backend_url

# --- Centralized Frontend Logging ---
def log_event_frontend(level, message, metadata=None, request_id="frontend"):
    try:
        payload = {
            "level": level,
            "message": message,
            "metadata": {
                "request_id": request_id,
                "source": "frontend",
                **(metadata or {})
            }
        }
        requests.post(f"{API_URL}/log", json=payload)
    except Exception as e:
        print(f"[LOGGING FAILED] {e}")


def search_and_clear(text_input, image):
    results, metadata = search(text_input, image)
    # Clear image after search
    return results, metadata, None


# --- Main Search Handler ---
def search(text_input, image):
    if not text_input and image is None:
        log_event_frontend("warning", "No input provided", 
        {"text_input": text_input, 
        "image_provided": False})
        return None, {"error": "Provide either text or an image."}

    request_id = datetime.utcnow().isoformat()
    headers = {"x-request-id": request_id}
    log_event_frontend("info", "Search initiated", {
        "request_id": request_id,
        "text_input": bool(text_input),
        "image_input": bool(image)
    })

    try:
        result_images = []
        result_metadata = []

        if image:
            log_event_frontend("info", "Sending image to backend", {"request_id": request_id})
            with open(image, "rb") as img_file:
                files = {"image": img_file}
                res = requests.post(f"{API_URL}/search/image", files=files, headers=headers)

        elif text_input:
            log_event_frontend("info", "Sending text to backend", {"request_id": request_id})
            res = requests.post(f"{API_URL}/search/text", data={"query": text_input}, headers=headers)

        else:
            return None, {"error": "No valid input received."}

        if not res.ok:
            log_event_frontend("error", "Backend search failed", {
                "request_id": request_id,
                "status": res.status_code
            })
            return None, {"error": "Backend search failed."}

        results = res.json().get("results", [])
        if not results:
            return [], {"error": "No results found."}

        for result in results:
            image_base64 = result.get("image")
            if not image_base64:
                continue

            image_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes))

            meta = result.get("metadata") or result
            caption = "\n".join(f"{k}: {v}" for k, v in meta.items())

            result_images.append((pil_image, caption))
            result_metadata.append(meta)

        log_event_frontend("info", "Returning all matches", {
            "request_id": request_id,
            "count": len(result_images)
        })
        return result_images, result_metadata

    except Exception as e:
        log_event_frontend("error", "Exception during search", {"request_id": request_id, "exception": str(e)})
        return None, {"error": str(e)}

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin: 20px 0;">
            🔍 ProductMatch Search
        </div>
        """,
        elem_id="title"
    )

    with gr.Row():
        with gr.Column(scale=3):
            result_gallery = gr.Gallery(
                label="Search Results",
                columns=3,
                height=750,
                object_fit="contain",
                preview=False
            )
            submit_button = gr.Button("Search", size="lg")

        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Describe the product...",
                lines=1
            )
            image_input = gr.Image(
                type="filepath",
                label="Upload an Image",
                height=400
            )
            result_metadata = gr.JSON(
                label="Metadata for Results",
                elem_id="metadata-box",
                height=250
            )

    submit_button.click(
        fn=search_and_clear,
        inputs=[text_input, image_input],
        outputs=[result_gallery, result_metadata, image_input]
    )

demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
