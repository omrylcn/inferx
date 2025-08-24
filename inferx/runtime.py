"""Main runtime interface for InferX"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .inferencers.base import BaseInferencer
from .inferencers.onnx_inferencer import ONNXInferencer
from .inferencers.openvino_inferencer import OpenVINOInferencer
from .inferencers.yolo import YOLOInferencer
from .inferencers.yolo_openvino import YOLOOpenVINOInferencer
from .settings import get_inferx_settings
from .recovery import with_model_loading_retry, with_inference_retry
from .exceptions import ModelError, InferenceError, ErrorCode


class InferenceEngine:
    """Main inference engine for InferX
    
    This class provides a high-level interface for running inference
    with different model types and runtimes.
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        runtime: str = "auto",
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the inference engine
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (yolo, anomalib, etc.). Auto-detected if None
            config: Configuration dictionary
            device: Device to run on (auto, cpu, gpu)
            runtime: Runtime to use (auto, onnx, openvino)
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.device = device
        self.runtime = runtime
        
        # Load global configuration with Pydantic validation
        self.global_config = get_inferx_settings()
        
        # Detect model type using configuration
        self.model_type = model_type or self.global_config.detect_model_type(
            self.model_path, runtime
        )
        
        # Build final configuration
        self.config = self._build_config(config)
        
        self.inferencer = self._create_inferencer()
    
    def _build_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build final configuration by merging defaults with user config
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Final merged configuration
        """
        # Start with model-specific defaults from global config
        config = self.global_config.get_model_defaults(self.model_type).copy()
        
        # Add preprocessing defaults based on runtime
        detected_runtime = self._detect_runtime()
        preprocessing_defaults = self.global_config.get_preprocessing_defaults(detected_runtime)
        if preprocessing_defaults:
            config.setdefault("preprocessing", {}).update(preprocessing_defaults)
        
        # Add runtime settings
        mapped_device = self.global_config.get_device_name(self.device)
        config.update({
            "device": mapped_device,
            "runtime": detected_runtime
        })
        
        # Merge with user configuration
        if user_config:
            self._merge_config(config, user_config)
        
        return config
    
    def _detect_runtime(self) -> str:
        """Detect runtime based on model type and user preference"""
        if self.runtime != "auto":
            return self.runtime
        
        # Auto-detect based on model type
        if "openvino" in self.model_type:
            return "openvino"
        elif "onnx" in self.model_type or self.model_path.suffix.lower() == '.onnx':
            return "onnx"
        elif self.model_path.suffix.lower() == '.xml':
            return "openvino"
        else:
            return "onnx"  # Default fallback
    
    def _merge_config(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target config"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    @with_model_loading_retry()
    def _create_inferencer(self) -> BaseInferencer:
        """Create appropriate inferencer based on model type with retry and fallback"""
        try:
            if self.model_type == "yolo_onnx":
                return YOLOInferencer(self.model_path, self.config)
            elif self.model_type == "yolo_openvino":
                return YOLOOpenVINOInferencer(self.model_path, self.config)
            elif self.model_type == "openvino":
                return OpenVINOInferencer(self.model_path, self.config)
            elif self.model_type == "onnx" or self._detect_runtime() == "onnx":
                return ONNXInferencer(self.model_path, self.config)
            elif self._detect_runtime() == "openvino":
                return OpenVINOInferencer(self.model_path, self.config)
            else:
                # Fallback to ONNX for unsupported types
                return ONNXInferencer(self.model_path, self.config)
        except Exception as e:
            # If all attempts fail, raise a meaningful error
            raise ModelError(
                message=f"Failed to create inferencer for model type '{self.model_type}'",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
                suggestions=[
                    "Check if model file exists and is readable",
                    "Verify model format is supported",
                    "Try different runtime (onnx/openvino)",
                    "Check system requirements and dependencies"
                ],
                recovery_actions=[
                    "Use auto model type detection",
                    "Convert model to different format",
                    "Check available runtimes"
                ],
                original_error=e,
                context={
                    "model_path": str(self.model_path),
                    "model_type": self.model_type,
                    "runtime": self.runtime,
                    "device": self.device
                }
            )
    
    @with_inference_retry()
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run inference on single input with retry and error recovery
        
        Args:
            input_data: Input data (image path, array, etc.)
            
        Returns:
            Inference results
            
        Raises:
            InferenceError: If inference fails after all retries
        """
        return self.inferencer.predict(input_data)
    
    def predict_batch(self, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """Run inference on batch of inputs
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of inference results
        """
        return self.inferencer.predict_batch(input_batch)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata
        
        Returns:
            Model information dictionary
        """
        info = self.inferencer.get_model_info()
        info.update({
            "model_type": self.model_type,
            "device": self.device,
            "runtime": self.runtime
        })
        return info