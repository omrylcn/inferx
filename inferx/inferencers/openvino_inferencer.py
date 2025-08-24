"""OpenVINO Runtime inferencer implementation"""

import openvino as ov
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .base import BaseInferencer
from ..utils import ImageProcessor, preprocess_for_inference


logger = logging.getLogger(__name__)

# Device mapping for OpenVINO
DEVICE_MAP = {
    "auto": "AUTO",
    "cpu": "CPU",
    "gpu": "GPU",
    "myriad": "MYRIAD",
    "hddl": "HDDL",
    "npu": "NPU"
}


class OpenVINOInferencer(BaseInferencer):
    """Generic OpenVINO Runtime inferencer for any OpenVINO model
    
    This class provides basic OpenVINO model loading and inference capabilities.
    It can be used directly for simple models or extended for model-specific logic.
    """
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """Initialize OpenVINO inferencer
        
        Args:
            model_path: Path to OpenVINO model file (.xml)
            config: Optional configuration dictionary
        """
        self.core = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.input_shape = None
        self.output_shape = None
        
        # Default configuration
        self.default_config = {
            "device": "AUTO",  # AUTO, CPU, GPU, MYRIAD, etc.
            "precision": "FP16",  # FP16, FP32, INT8
            "num_threads": 0,  # 0 = auto, >0 = specific thread count
            "num_streams": 0,  # 0 = auto, >0 = specific stream count
            "cache_dir": None,  # Model cache directory
            "performance_hint": "THROUGHPUT",  # THROUGHPUT, LATENCY, CUMULATIVE_THROUGHPUT
            "preprocessing": {
                "target_size": [224, 224],  # Default input size
                "normalize": True,
                "color_format": "RGB",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        
        # Merge with user config
        if config:
            self._merge_config(self.default_config, config)
        
        super().__init__(model_path, self.default_config)
    
    def _merge_config(self, default: Dict, user: Dict) -> None:
        """Recursively merge user config with default config"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _load_model(self) -> None:
        """Load OpenVINO model using OpenVINO Runtime"""
        try:
            # Initialize OpenVINO Core
            self.core = ov.Core()
            
            # Check if model file has .xml extension
            model_path = Path(self.model_path)
            if model_path.suffix.lower() != '.xml':
                raise ValueError(f"OpenVINO model must have .xml extension, got: {model_path.suffix}")
            
            # Check if .bin file exists
            bin_path = model_path.with_suffix('.bin')
            if not bin_path.exists():
                logger.warning(f"Binary file not found: {bin_path}. Model may be embedded in XML.")
            
            # Load model
            model = self.core.read_model(str(model_path))
            
            # Setup device configuration
            device_config = self._setup_device_config()
            
            # Get device name from config and map it to OpenVINO device name
            device = self.config.get("device", "AUTO")
            openvino_device = DEVICE_MAP.get(device.lower(), device.upper())
            
            # Compile model for target device
            logger.info(f"Compiling model for device: {openvino_device}")
            logger.info(f"Device configuration: {device_config}")
            
            self.compiled_model = self.core.compile_model(
                model=model,
                device_name=openvino_device,
                config=device_config
            )
            
            # Get model metadata
            self._extract_model_metadata()
            
            logger.info(f"Loaded OpenVINO model: {self.model_path}")
            logger.info(f"Device: {openvino_device}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model {self.model_path}: {e}")
            raise RuntimeError(f"Failed to load OpenVINO model: {e}")
    
    def _setup_device_config(self) -> Dict[str, Any]:
        """Setup device-specific configuration"""
        config = {}
        
        # Get device name from config and map it to OpenVINO device name
        device = self.config.get("device", "AUTO")
        openvino_device = DEVICE_MAP.get(device.lower(), device.upper())
        
        # Performance hint
        perf_hint = self.config.get("performance_hint", "THROUGHPUT")
        if perf_hint in ["THROUGHPUT", "LATENCY", "CUMULATIVE_THROUGHPUT"]:
            config["PERFORMANCE_HINT"] = perf_hint
        
        # Thread settings for CPU
        if openvino_device == "CPU":
            num_threads = self.config.get("num_threads", 0)
            if num_threads > 0:
                config["CPU_THREADS_NUM"] = str(num_threads)
            
            num_streams = self.config.get("num_streams", 0)
            if num_streams > 0:
                config["CPU_THROUGHPUT_STREAMS"] = str(num_streams)
        
        # GPU settings
        if openvino_device == "GPU":
            num_streams = self.config.get("num_streams", 0)
            if num_streams > 0:
                config["GPU_THROUGHPUT_STREAMS"] = str(num_streams)
        
        # Model caching
        cache_dir = self.config.get("cache_dir")
        if cache_dir:
            config["CACHE_DIR"] = str(cache_dir)
        
        return config
    
    def _extract_model_metadata(self) -> None:
        """Extract input/output metadata from loaded model"""
        # Get input/output info
        input_info = self.compiled_model.inputs
        output_info = self.compiled_model.outputs
        
        if len(input_info) == 0:
            raise RuntimeError("Model has no inputs")
        if len(output_info) == 0:
            raise RuntimeError("Model has no outputs")
        
        # Store primary input/output layers
        self.input_layer = input_info[0]
        self.output_layer = output_info[0]
        
        # Store shapes
        self.input_shape = list(self.input_layer.shape)
        self.output_shape = list(self.output_layer.shape)
        
        # Store all inputs/outputs for complex models
        self.input_layers = {inp.any_name: inp for inp in input_info}
        self.output_layers = {out.any_name: out for out in output_info}
        
        logger.debug(f"Model inputs: {list(self.input_layers.keys())}")
        logger.debug(f"Model outputs: {list(self.output_layers.keys())}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
    
    def preprocess(self, input_data: Any) -> np.ndarray:
        """Preprocess input data for OpenVINO model
        
        Args:
            input_data: Input data (image path, image array, etc.)
            
        Returns:
            Preprocessed data ready for model inference
        """
        if isinstance(input_data, (str, Path)):
            # Input is image path
            preprocessing_config = self.config.get("preprocessing", {})
            
            return preprocess_for_inference(
                image_path=input_data,
                target_size=preprocessing_config.get("target_size", [224, 224]),
                normalize=preprocessing_config.get("normalize", True),
                color_format=preprocessing_config.get("color_format", "RGB"),
                mean=tuple(preprocessing_config.get("mean", [0.485, 0.456, 0.406])),
                std=tuple(preprocessing_config.get("std", [0.229, 0.224, 0.225]))
            )
        
        elif isinstance(input_data, np.ndarray):
            # Input is already a numpy array
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Ensure batch dimension exists
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            
            return input_data
        
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run OpenVINO model inference
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            # Create inference request
            infer_request = self.compiled_model.create_infer_request()
            
            # Set input data
            if len(self.input_layers) == 1:
                # Single input model
                infer_request.set_input_tensor(preprocessed_data)
            else:
                # Multiple input model - basic implementation
                # More sophisticated logic needed for complex models
                primary_input_name = list(self.input_layers.keys())[0]
                infer_request.set_tensor(primary_input_name, preprocessed_data)
            
            # Run inference
            infer_request.infer()
            
            # Get outputs
            outputs = []
            for output_layer in self.output_layers.values():
                output_tensor = infer_request.get_output_tensor(output_layer.index)
                outputs.append(output_tensor.data.copy())
            
            logger.debug(f"Inference completed, output shapes: {[out.shape for out in outputs]}")
            return outputs
            
        except Exception as e:
            logger.error(f"OpenVINO inference failed: {e}")
            raise RuntimeError(f"OpenVINO inference failed: {e}")
    
    def postprocess(self, model_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Basic postprocessing for OpenVINO model outputs
        
        Args:
            model_outputs: Raw outputs from model inference
            
        Returns:
            Dictionary containing processed results
        """
        # Basic postprocessing - just return raw outputs with metadata
        results = {
            "raw_outputs": [output.tolist() for output in model_outputs],
            "output_shapes": [output.shape for output in model_outputs],
            "num_outputs": len(model_outputs),
            "model_type": "openvino_generic"
        }
        
        # Add some basic analysis
        output_names = list(self.output_layers.keys())
        for i, output in enumerate(model_outputs):
            output_name = output_names[i] if i < len(output_names) else f"output_{i}"
            results[f"{output_name}_shape"] = output.shape
            results[f"{output_name}_dtype"] = str(output.dtype)
            
            # Basic statistics
            if output.size > 0:
                results[f"{output_name}_min"] = float(np.min(output))
                results[f"{output_name}_max"] = float(np.max(output))
                results[f"{output_name}_mean"] = float(np.mean(output))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the OpenVINO model
        
        Returns:
            Dictionary containing model metadata
        """
        base_info = super().get_model_info()
        
        if self.compiled_model is None:
            return base_info
        
        # Add OpenVINO-specific information
        openvino_info = {
            "runtime": "openvino",
            "device": self.config.get("device", "AUTO"),
            "precision": self.config.get("precision", "FP16"),
            "input_names": list(self.input_layers.keys()) if hasattr(self, 'input_layers') else [],
            "output_names": list(self.output_layers.keys()) if hasattr(self, 'output_layers') else [],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }
        
        # Get device info if available
        try:
            if hasattr(self.core, 'get_property'):
                device_name = self.config.get("device", "AUTO")
                if device_name != "AUTO":
                    device_info = {
                        "device_name": self.core.get_property(device_name, "FULL_DEVICE_NAME"),
                    }
                    openvino_info.update(device_info)
        except Exception as e:
            logger.debug(f"Could not extract device info: {e}")
        
        base_info.update(openvino_info)
        return base_info
    
    def get_available_devices(self) -> List[str]:
        """Get list of available OpenVINO devices
        
        Returns:
            List of available device names
        """
        if self.core is None:
            self.core = ov.Core()
        
        return self.core.available_devices