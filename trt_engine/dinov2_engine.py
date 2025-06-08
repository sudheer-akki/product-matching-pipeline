import torch
import argparse
from trt_base import TRTEngineBuilder, TRTInferenceBase
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("DINOv2 TensorRT Script Started")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for DINOv2 model.")
    parser.add_argument('--model_name', type=str, default="dinov2", help='Name tag for the model')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--engine_path', type=str, required=True, help='Path to save the TensorRT engine')
    parser.add_argument('--dtype', type=str, default='fp16', choices=["fp16", "fp32"], help='Precision type')
    return parser.parse_args()


class DinoV2_TRT_Converter:
    def __init__(self, onnx_path: str, engine_path: str):
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.input_names = ["pixel_values"]
        self.output_names = ["x_norm_clstoken"]  # Make sure this matches your ONNX output

        # Only batch size is dynamic
        self.dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "x_norm_clstoken": {0: "batch_size"},
        }

        self.input_shapes = {
            "pixel_values": (
                (1, 3, 224, 224),  # min
                (4, 3, 224, 224),  # opt
                (8, 3, 224, 224)   # max
            )
        }

        self.trt_builder = TRTEngineBuilder(
            onnx_path=self.onnx_path,
            engine_path=self.engine_path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes
        )

    def build(self, precision="fp16"):
        logger.info("Building DINOv2 TensorRT engine...")
        self.trt_builder.build_engine(self.input_shapes, precision=precision)


if __name__ == "__main__":
    args = parse_arguments()

    print("Model Configuration:")
    print(f"ONNX_PATH: {args.onnx_path}")
    print(f"ENGINE_PATH: {args.engine_path}")
    print(f"DTYPE: {args.dtype}")

    try:
        converter = DinoV2_TRT_Converter(
            onnx_path=args.onnx_path,
            engine_path=args.engine_path
        )
        converter.build(precision=args.dtype)
        logger.info("DINOv2 TensorRT engine built successfully.")
    except Exception as e:
        logger.error(f"Failed to build DINOv2 TensorRT engine: {e}", exc_info=True)
