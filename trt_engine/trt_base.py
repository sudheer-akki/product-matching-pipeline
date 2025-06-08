# trt_base.py
import os
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import logging
logger = logging.getLogger(__name__)
logger.info("This is from TRT Base script")

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

class TRTEngineBuilder:
    def __init__(self, 
                 onnx_path: str,
                 engine_path: str,
                 input_names: list,
                 output_names: list,
                 dynamic_axes: dict,
                 trt_logger_verbosity=trt.Logger.INFO):
        
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes

        # TensorRT setup
        self.trt_logger = trt.Logger(trt_logger_verbosity)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 512 MB
        self.network = None
        self.parser = None


    def build_engine(self, input_shapes: dict, precision: str = "fp16"):
        logger.info("Building TensorRT engine from ONNX...")
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        with open(self.onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                for i in range(self.parser.num_errors):
                    logger.error(f"Parser error: {self.parser.get_error(i)}")
                raise RuntimeError("ONNX parsing failed.")

        if precision in ("fp16", "float16") and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled.")
        elif precision == "int8" and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 mode enabled.")

        profile = self.builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
            logger.info(f"Optimization profile for {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(profile)

        engine = self.builder.build_serialized_network(self.network, self.config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")

        os.makedirs(self.engine_path, exist_ok=True)
        engine_file = os.path.join(self.engine_path, "model.plan")
        with open(engine_file, "wb") as f:
            f.write(engine)

        logger.info(f" TensorRT engine saved to: {engine_file}")


class TRTInferenceBase:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self._buffers_allocated = False

    def _load_engine(self, path):
        try:
            with open(path, "rb") as f:
                logger.info(f"Loading TensorRT engine from {path}")
                return self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            logger.error(f"Failed to load engine: {e}", exc_info=True)
            raise

    def _allocate_buffers(self):
        inputs, outputs, bindings = {}, {}, []
        stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            print("shape",shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = 1 #trt.volume(shape) #if -1 not in shape else 1  # Support for dynamic shapes
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(0)  # placeholder

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = HostDeviceMem(host_mem, device_mem)
            else:
                outputs[name] = HostDeviceMem(host_mem, device_mem)

        return inputs, outputs, bindings, stream

    def infer(self, input_tensors: dict, output_names: list):
        if not self._buffers_allocated:
            # After set_input_shape
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
            self._buffers_allocated = True
        try:
            # Set input shapes for dynamic shapes
            for name, tensor in input_tensors.items():
                #tensor = tensor.astype(np.int32)
                if tensor.dtype != np.int64:
                    tensor = tensor.astype(np.int64)
                self.context.set_input_shape(name, tensor.shape)

                np.copyto(self.inputs[name].host, tensor.ravel())
                cuda.memcpy_htod_async(self.inputs[name].device, self.inputs[name].host, self.stream)

            # Set addresses for all inputs/outputs
            for i, name in enumerate(self.engine):
                if name in self.inputs:
                    self.context.set_tensor_address(name, int(self.inputs[name].device))
                    self.bindings[i] = int(self.inputs[name].device)
                elif name in self.outputs:
                    self.context.set_tensor_address(name, int(self.outputs[name].device))
                    self.bindings[i] = int(self.outputs[name].device)

            # Execute inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Copy outputs back
            results = {}
            for name in output_names:
                cuda.memcpy_dtoh_async(self.outputs[name].host, self.outputs[name].device, self.stream)
            self.stream.synchronize()

            for name in output_names:
                shape = self.context.get_tensor_shape(name)
                results[name] = self.outputs[name].host.reshape(shape)
            logger.info(f"Inference completed")
            return results
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
