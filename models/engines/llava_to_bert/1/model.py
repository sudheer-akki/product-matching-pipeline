import os
import io
import yaml
import numpy as np
from PIL import Image, UnidentifiedImageError
import triton_python_backend_utils as pb_utils

# LLaVA pipeline imports
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner

# HugginFace BERT
from transformers import BertTokenizer
from torch.utils.dlpack import from_dlpack

class TritonPythonModel:
    def initialize(self, args):
        # Load configuration from YAML
        config_path = "/workspace/configs/app_config.yaml"
        config = self._load_config(config_path)
        llava_config = config['models']['llava']
        # --- LLaVA engine setup ---
        self.llava_args = type('', (), {})()
        self.llava_args.max_new_tokens = 60
        self.llava_args.hf_model_dir = llava_config['hf_model_dir']
        self.llava_args.visual_engine_dir = llava_config['visual_engine_dir']
        self.llava_args.llm_engine_dir = llava_config['llm_engine_dir']
        self.llava_args.visual_engine_name = llava_config.get('visual_engine_name', "model.engine")
        self.llava_args.llm_engine_name = llava_config.get('llm_engine_name', "rank0.engine")
        self.llava_args.log_level = llava_config.get('log_level', "info")
        self.llava_args.input_text = ""
        self.llava_args.batch_size = 1
        self.llava_args.num_beams = 1
        self.llava_args.top_k = 1
        self.llava_args.top_p = 0.0
        self.llava_args.temperature = 1.0
        self.llava_args.repetition_penalty = 1.0
        self.llava_args.run_profiling = False
        self.llava_args.profiling_iterations = 1
        self.llava_args.check_accuracy = False
        self.llava_args.image_path = None
        self.llava_args.path_sep = ","
        self.llava_args.video_path = None
        self.llava_args.video_num_frames = None
        self.llava_args.enable_context_fmha_fp32_acc = False
        self.llava_args.enable_chunked_context = False
        self.llava_args.use_py_session = False
        self.llava_args.kv_cache_free_gpu_memory_fraction = 0.9
        self.llava_args.cross_kv_cache_fraction = 0.5
        self.llava_args.multi_block_mode = True

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        logger.set_level(self.llava_args.log_level)
        self.llava_model = MultimodalModelRunner(self.llava_args)
        print("LLaVA runner loaded", flush=True)

        # --- BERT embedding setup (change model name if needed) ---
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model_name = "bert"
        print("BERT Tokenizer loaded", flush=True)

    @staticmethod
    def _load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def decode_image_bytes(image_bytes):
        # Defensive: sometimes you get np.ndarray of shape (1,) and dtype object, so fix that.
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.item() if image_bytes.size == 1 else image_bytes[0]
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError(f"Expected bytes, got {type(image_bytes)}")
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            print("[ERROR] decode_image_bytes: UnidentifiedImageError, cannot decode as image")
            return None
        except Exception as e:
            print(f"[ERROR] decode_image_bytes: {e}")
            return None

    def execute(self, requests):
        print("===== TritonPythonModel.execute called =====", flush=True)
        print(f"[EXECUTE] Number of requests: {len(requests)}", flush=True)
        responses = []

        for request in requests:
            # 1. Extract batched inputs
            print("[EXECUTE] Extracting inputs...", flush=True)
            images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()  # [B]
            prompts = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()  # [B]
            max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()
            batch_size = images.shape[0]
            print(f"[INFO] Batch size: {batch_size}", flush=True)

            print(f"[INFO] Formatted prompts: {prompts}", flush=True)
            print(f"[INFO] Batch size: {batch_size}", flush=True)
            print(f"[INFO] images dtype: {images.dtype}, shape: {images.shape}", flush=True)
            print(f"[INFO] max_tokens: {max_tokens}", flush=True)

            output_captions = []
            output_embeddings = []

            for i in range(batch_size):
                image_bytes = images[i]
                prompt = prompts[i]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")

                max_new_tokens = int(max_tokens[i]) if max_tokens is not None else 30

                pil_img = self.decode_image_bytes(image_bytes)
                if pil_img is None:
                    caption = "Error: Unable to decode image"
                    output_captions.append(caption)
                    output_embeddings.append(np.zeros((768,), dtype=np.float32))  # fallback vector
                    continue

                self.llava_args.input_text = prompt
                self.llava_args.max_new_tokens = max_new_tokens
                try:
                    _, output_text = self.llava_model.run(
                        self.llava_args.input_text, pil_img, self.llava_args.max_new_tokens
                        )
                    caption = output_text[0][0] if isinstance(output_text[0], list) else output_text[0]
                except Exception as e:
                    caption = f"Error: {str(e)}"

                print(f"[INFO] Received captions from LLaVA: {caption}", flush=True)

                tokens = self.bert_tokenizer(
                    caption,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )
                print(f"[INFO] Tokens input_ids shape: {tokens['input_ids'].shape}, \
                attention_mask shape: {tokens['attention_mask'].shape}", flush=True)

                # 4. Send batched BERT request
                bert_inputs = [
                    pb_utils.Tensor("input_ids", tokens["input_ids"].astype(np.int64)),
                    pb_utils.Tensor("attention_mask", tokens["attention_mask"].astype(np.int64))
                ]

                print(f"[INFO] Inputs for the BERT model: {bert_inputs}", flush=True)
                print(f"[INFO] Sending request to BERT model: {self.bert_model_name}", flush=True)
                bert_request = pb_utils.InferenceRequest(
                    model_name=self.bert_model_name,
                    inputs=bert_inputs,
                    requested_output_names=["pooled_output"]
                )
                bert_response = bert_request.exec()
                embeddings_tensor = pb_utils.get_output_tensor_by_name(bert_response, "pooled_output")
                if embeddings_tensor is None:
                    print("[ERROR] No 'pooled_output' in BERT response!", flush=True)
                    print("[ERROR] All response tensors:", [t.name() for t in bert_response], flush=True)
                    raise ValueError("BERT did not return 'pooled_output'")
                try:
                    if embeddings_tensor.is_cpu():
                        embeddings_np = embeddings_tensor.as_numpy()
                    else:
                        embeddings_np = from_dlpack(embeddings_tensor.to_dlpack()).to("cpu").cpu().detach().numpy()
                except Exception as e:
                    print(f"[ERROR] BERT Triton inference failed: {e}", flush=True)
                    embeddings_np = np.zeros((1, 768), dtype=np.float32)
                    
                output_captions.append(caption)
                output_embeddings.append(embeddings_np[0])

        out_caption = pb_utils.Tensor("text", np.array(output_captions, dtype=np.object_))
        out_embed = pb_utils.Tensor("pooled_output", np.stack(output_embeddings).astype(np.float32))
        responses.append(pb_utils.InferenceResponse(output_tensors=[out_caption, out_embed]))
        print("===== TritonPythonModel.execute completed =====", flush=True)
        return responses
