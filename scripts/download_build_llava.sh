#!/bin/bash

# Exit on error
set -e

# Define model name
export MODEL_NAME="llava-1.5-7b-hf"

# Create workspace if needed
mkdir -p models/llava

# Clone model from Hugging Face
git clone https://huggingface.co/llava-hf/${MODEL_NAME} models/${MODEL_NAME}

# Download convert_checkpoint.py from GitHub
curl -L https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.10.0/examples/llama/convert_checkpoint.py -o convert_checkpoint.py

# Run checkpoint conversion
python3 convert_checkpoint.py --model_dir models/${MODEL_NAME}

# Build TensorRT-LLM engine
trtllm-build \
    --checkpoint_dir "workspace/tllm_checkpoint" \
    --output_dir "models/1" \
    --gemm_plugin float16 \
    --max_batch_size 2 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_multimodal_len 4608 \
    > llava_engine_build.log 2>&1
