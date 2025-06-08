import argparse
import logging
from trt_base import TRTEngineBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create TensorRT engine for BERT from ONNX.")
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--engine_path', type=str, required=True, help='Path to save TensorRT engine dir')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32'], help='Precision to use')
    parser.add_argument('--min_length', type=int, default=128, help='Minimum sequence length')
    parser.add_argument('--opt_length', type=int, default=128, help='Optimal sequence length')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    return parser.parse_args()

def main():
    args = parse_arguments()

    logger.info("Creating BERT TensorRT engine...")

    input_shapes = {
        "input_ids": ((1, args.min_length), (1, args.opt_length), (1, args.max_length)),
        "attention_mask": ((1, args.min_length), (1, args.opt_length), (1, args.max_length)),
    }

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "last_hidden_state": {0: "batch_size"}
    }

    engine_builder = TRTEngineBuilder(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes
    )

    engine_builder.build_engine(input_shapes, precision=args.dtype)

    logger.info("✓ BERT TensorRT engine created successfully.")

if __name__ == "__main__":
    main()
