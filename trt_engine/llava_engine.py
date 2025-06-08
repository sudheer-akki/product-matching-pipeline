import os
import subprocess
import argparse
import shutil
from llava_utils import convert_checkpoint
import logging
import importlib
logger = logging.getLogger(__name__)

logger.info("This is from LLAVA script")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure LLaVA model parameters.")
    parser.add_argument('--model_name', type=str, default="llava-1.5-7b-hf", help='Name of the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model source directory')
    parser.add_argument('--unified_ckpt_path', type=str, required=True, help='Path to the unified checkpoint directory')
    parser.add_argument('--engine_dir', type=str, required=True, help='Path to the TensorRT engine plan file')
    parser.add_argument('--dtype', type=str, default='float16', help='Precision type (e.g., float16, float32)')
    return parser.parse_args()

def install_tensorrt_llm_python(version: str = "0.16.0"):
    repo_url = "https://github.com/triton-inference-server/tensorrtllm_backend.git"
    repo_dir = "tensorrtllm_backend"

    # Clone the repo
    subprocess.run(["git", "clone", "--branch", version, repo_url], check=True)

    # Install git-lfs
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "git-lfs"], check=True)
    subprocess.run(["git", "lfs", "install"], check=True)

    # Init submodules inside the repo
    os.chdir(repo_dir)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)


def download_llava_model_with_git(target_dir, model_name):
    repo_url = f"https://huggingface.co/llava-hf/{model_name}"
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        logger.info(f"Model not found at '{target_dir}'. Cloning from {repo_url}...")
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
        shutil.copytree(target_dir, model_name, dirs_exist_ok=True)
    else:
        logger.info("Model directory already exists. Skipping download.")

def convert_checkpoint_if_needed(model_dir, output_dir, dtype):
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        logger.info(f"No checkpoints found in '{output_dir}'. Starting checkpoint conversion...")
        convert_checkpoint(model_dir=model_dir, output_dir=output_dir, dtype=dtype)
        logger.info("Checkpoint conversion completed.")
    else:
        logger.info("Checkpoint already exists. Skipping conversion.")

def run_trtllm_build(checkpoint_dir, output_dir):
    logger.info(f"Building TRT-LLM engine from checkpoint at '{checkpoint_dir}'...")
    command = [
        "trtllm-build",
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", output_dir,
        "--gemm_plugin", "float16",
        "--use_fused_mlp",
        "--max_batch_size", "2",
        "--max_input_len", "64",
        "--max_output_len", "128",
        "--max_multimodal_len", "512"
    ]
    subprocess.run(command, check=True)
    logger.info(f"TRT-LLM engine built successfully at '{output_dir}'.")

def ensure_tensorrt_llm_installed(version= "0.16.0"):
    print("🔍 Checking for tensorrt_llm Python package...")
    if importlib.util.find_spec("tensorrt_llm") is not None:
        print("tensorrt_llm is already installed.")
    else:
        print(" tensorrt_llm not found. Installing from source...")
        install_tensorrt_llm_python(version=version)

if __name__ == "__main__":
    args = parse_arguments()

    MODEL_NAME = args.model_name
    MODEL_PATH = args.model_path
    UNIFIED_CKPT_PATH = args.unified_ckpt_path
    ENGINE_DIR = args.engine_dir
    DTYPE = args.dtype

    print("LLaVA Model Configuration:")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"UNIFIED_CKPT_PATH: {UNIFIED_CKPT_PATH}")
    print(f"ENGINE_DIR: {ENGINE_DIR}")
    print(f"DTYPE: {DTYPE}")

    "Checking Tensorrt LLM Python API"
    ensure_tensorrt_llm_installed(version="0.16.0")

    logger.info("Starting LLaVA model preparation pipeline...")

    download_llava_model_with_git(target_dir=MODEL_PATH,
                                  model_name=MODEL_NAME)
    
    convert_checkpoint_if_needed(model_dir = MODEL_PATH, 
                                output_dir = UNIFIED_CKPT_PATH, 
                                dtype = DTYPE)

    run_trtllm_build(checkpoint_dir=UNIFIED_CKPT_PATH, 
                     output_dir= ENGINE_DIR)
    
    logger.important("Llava Engine completed successfully.")