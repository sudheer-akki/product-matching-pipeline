#!/bin/bash
set -e

read -p "Build BERT TensorRT engine? (y/n): " build_bert
if [[ "$build_bert" == "y" || "$build_bert" == "Y" ]]; then
  echo "Building BERT TensorRT engine..."
  python3 workspace/trt_engines/bert_engine.py \
    --onnx_path ./models/engines/bert/bert.onnx \
    --engine_path ./models/engines/bert/1/model.plan \
    --dtype fp16 \
    --max_length 128 \
    --opt_length 64 \
    --min_length 16
fi

read -p "Build the DINOv2 TensorRT engine? (y/n): " build_dinov2
if [[ "$build_dinov2" == "y" || "$build_dinov2" == "Y" ]]; then
  echo "Building DINOv2 TensorRT engine..."
  python3 trt_engines/dinov2_engine.py \
    --onnx_path ./models/engines/dinov2/dinov2.onnx \
    --engine_path ./models/engines/dinov2/1/model.plan \
    --dtype fp16
fi

read -p "Build the LLaVA TensorRT engine? (y/n): " build_llava
if [[ "$build_llava" == "y" || "$build_llava" == "Y" ]]; then
  HF_LLAVA_MODEL=llava-1.5-7b-hf
  UNIFIED_CKPT_PATH=llava-1.5-7b-ckpt

  echo "Building LLaVA TensorRT engine..."
  python3 trt_engines/llava_engine.py \
    --model_path ./models/model_src/llava/${HF_LLAVA_MODEL} \
    --unified_ckpt_path ./models/model_src/llava/${UNIFIED_CKPT_PATH} \
    --engine_dir ./models/engines/llava/1/llava.plan \
    --dtype float16
fi

echo "Selected TensorRT engine build(s) completed."
