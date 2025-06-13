#!/bin/bash
set -e

# ====== BERT Variables ======
BERT_ONNX_PATH="./models/engines/bert/bert.onnx"
BERT_ENGINE_PATH="./models/engines/bert/1/model.plan"
BERT_DTYPE="fp16"
BERT_MAX_LENGTH=128
BERT_OPT_LENGTH=64
BERT_MIN_LENGTH=16

# ====== DINOV2 Variables ======
DINOV2_ONNX_PATH="./models/engines/dinov2/dinov2.onnx"
DINOV2_ENGINE_PATH="./models/engines/dinov2/1/model.plan"
DINOV2_DTYPE="fp16"

read -p "Build BERT TensorRT engine? (y/n): " build_bert
if [ -f "${BERT_ENGINE_PATH}" ]; then
  echo "BERT TensorRT engine already exists at ${BERT_ENGINE_PATH}, skipping build."
else
  echo "Building BERT TensorRT engine..."
  python3 trt_engines/bert_engine.py \
    --onnx_path "${BERT_ONNX_PATH}" \
    --engine_path "${BERT_ENGINE_PATH}" \
    --dtype "${BERT_DTYPE}" \
    --max_length ${BERT_MAX_LENGTH} \
    --opt_length ${BERT_OPT_LENGTH} \
    --min_length ${BERT_MIN_LENGTH}
fi

read -p "Build the DINOv2 TensorRT engine? (y/n): " build_dinov2
if [ -f "${DINOV2_ENGINE_PATH}" ]; then
  echo "DINOv2 TensorRT engine already exists at ${DINOV2_ENGINE_PATH}, skipping build."
else
  echo "Building DINOv2 TensorRT engine..."
  python3 trt_engines/dinov2_engine.py \
    --onnx_path "${DINOV2_ONNX_PATH}" \
    --engine_path "${DINOV2_ENGINE_PATH}" \
    --dtype "${DINOV2_DTYPE}"
fi

# ====== Llava-Vision Variables ======
RELEASE_BRANCH="0.16.0"
MODEL_NAME="llava-onevision-qwen2-7b-ov-hf"
MODEL_REPO="llava-hf/${MODEL_NAME}"
MODEL_DIR="./models/${MODEL_NAME}"
CKPT_DIR="./models/ckpt/"
LLAVA_VISION_DIR="./models/llava_vision/"
LLAVA_VISION_ENCODER_DIR="./models/llava_vision_encoder/"
SAMPLE_IMAGE="./dataset/sample/MEN-Denim-id_00000089-08_4_full.jpg"

# ====== Cloning tensorrt_llm backend ======
if [ -d "tensorrtllm_backend" ]; then
  echo "tensorrtllm_backend folder already exists, skipping clone."
else
  git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch ${RELEASE_BRANCH}
  cd tensorrtllm_backend
  apt-get update && apt-get install -y git-lfs --no-install-recommends
  git lfs install
  git submodule update --init --recursive
  cd ..
fi

# ====== Downloading Llava-Vision Model ======
if [ -d "${MODEL_DIR}" ]; then
  echo "Model directory ${MODEL_DIR} already exists, skipping download."
else
  git clone https://huggingface.co/${MODEL_REPO} ${MODEL_DIR}
fi


# ====== Converting Checkpoints ======
if [ -d "${CKPT_DIR}" ] && [ "$(ls -A ${CKPT_DIR})" ]; then
  echo "Checkpoint directory ${CKPT_DIR} already exists, skipping conversion."
else
  python3 ./tensorrtllm_backend/tensorrt_llm/examples/qwen/convert_checkpoint.py \
    --model_dir ${MODEL_DIR} \
    --output_dir ${CKPT_DIR} \
    --dtype float16
fi

# ====== Build LLava Vision Engine ======
if [ -d "${LLAVA_VISION_DIR}" ] && [ "$(ls -A ${LLAVA_VISION_DIR})" ]; then
  echo "LLava vision engine directory ${LLAVA_VISION_DIR} already exists, skipping build."
else
  trtllm-build \
    --checkpoint_dir ${CKPT_DIR} \
    --output_dir ${LLAVA_VISION_DIR} \
    --gemm_plugin float16 \
    --use_fused_mlp=enable \
    --max_batch_size 1 \
    --max_input_len 7228 \
    --max_seq_len 7328 \
    --max_multimodal_len 7128 \
    --max_prompt_embedding_table_size 8192
fi

# ====== Building TensorRT engines for visual components ======
if [ -d "${LLAVA_VISION_ENCODER_DIR}" ] && [ "$(ls -A ${LLAVA_VISION_ENCODER_DIR})" ]; then
  echo "LLava vision encoder directory ${LLAVA_VISION_ENCODER_DIR} already exists, skipping build."
else
  python3 ./tensorrtllm_backend/tensorrt_llm/examples/multimodal/build_visual_engine.py \
    --model_path ${MODEL_DIR} \
    --model_type llava_onevision \
    --output_dir ${LLAVA_VISION_ENCODER_DIR} \
    --max_batch_size 8
fi


read -p "Run Llava inference on sample image? (y/n): " run_llava_infer
if [[ "$run_llava_infer" == "y" || "$run_llava_infer" == "Y" ]]; then
  python3 ./tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py \
    --max_new_tokens 30 \
    --hf_model_dir ${MODEL_DIR} \
    --visual_engine_dir ${LLAVA_VISION_ENCODER_DIR} \
    --llm_engine_dir ${LLAVA_VISION_DIR} \
    --input_text "What is shown in this image?" \
    --image_path ${SAMPLE_IMAGE}
fi
fi

echo "Selected TensorRT engine build(s) completed."
