"""YOLO object detection inferencer using OpenVINO Runtime"""

import openvino as ov
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import logging

from .yolo_base import BaseYOLOInferencer
from ..utils import ImageProcessor
from ..exceptions import ModelError, ErrorCode

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


class YOLOOpenVINOInferencer(BaseYOLOInferencer):
    """YOLO object detection inferencer optimized for OpenVINO Runtime"""
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict] = None):
        """Initialize YOLO OpenVINO inferencer
        
        Args:
            model_path: Path to YOLO OpenVINO model file (.xml)
            config: Optional configuration dictionary
        """
        # Default YOLO config optimized for OpenVINO
        default_config = {
            "device": "AUTO",  # Let OpenVINO choose best device
            "performance_hint": "THROUGHPUT",  # Optimize for throughput
            "precision": "FP16",  # Use FP16 for better performance
            "input_size": 640,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "max_detections": 100,
            "class_names": [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        }
        
        # Merge user config with YOLO-specific defaults
        if config:
            default_config.update(config)
        
        # Initialize OpenVINO components before calling super().__init__
        self.core = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.input_shape = None
        self.output_shape = None
        
        super().__init__(model_path, default_config)
    
    def _setup_device_config(self) -> Dict[str, Any]:
        """Setup OpenVINO device configuration optimized for YOLO"""
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
        # Handle tensors that may not have names
        self.input_layers = {}
        self.output_layers = {}
        
        for i, inp in enumerate(input_info):
            try:
                name = inp.any_name
            except RuntimeError:
                name = f"input_{i}"
            self.input_layers[name] = inp
            
        for i, out in enumerate(output_info):
            try:
                name = out.any_name
            except RuntimeError:
                name = f"output_{i}"
            self.output_layers[name] = out
            
        logger.debug(f"Model inputs: {list(self.input_layers.keys())}")
        logger.debug(f"Model outputs: {list(self.output_layers.keys())}")
        logger.debug(f"Input shape: {self.input_shape}")
        logger.debug(f"Output shape: {self.output_shape}")
    
    def _run_inference(self, preprocessed_data: np.ndarray) -> List[np.ndarray]:
        """Run YOLO model inference on OpenVINO
        
        Args:
            preprocessed_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            # Run inference directly using compiled model
            # This approach mimics the Ultralytics OpenVINO integration
            result = self.compiled_model([preprocessed_data])
            
            # Handle OVDict result type
            if hasattr(result, 'keys'):
                # OVDict type - get first (and likely only) output
                output_key = list(result.keys())[0]
                output_tensor = result[output_key]
                output_array = np.array(output_tensor)
                outputs = [output_array]
            else:
                # Other result types
                outputs = [np.array(result)]
            
            logger.debug(f"YOLO OpenVINO inference completed, output shapes: {[out.shape for out in outputs]}")
            return outputs
            
        except Exception as e:
            logger.error(f"YOLO OpenVINO inference failed: {e}")
            raise RuntimeError(f"YOLO OpenVINO inference failed: {e}")
    
    def preprocess(self, input_data: Any) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """YOLO preprocessing optimized for OpenVINO
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Tuple of (preprocessed_image, scale_ratio, padding)
        """
        # Store original image dimensions for postprocessing
        if isinstance(input_data, (str, Path)):
            # Load image from file to get dimensions
            img = ImageProcessor.load_image(input_data)
            self.img_height, self.img_width = img.shape[:2]
        elif isinstance(input_data, np.ndarray):
            self.img_height, self.img_width = input_data.shape[:2]
        
        return self.common_preprocess(input_data)
    
    def postprocess(self, model_outputs: List[np.ndarray], ratio: Tuple[float, float], pad: Tuple[int, int]) -> Dict[str, Any]:
        """YOLO postprocessing: NMS + format results
        
        Args:
            model_outputs: Raw outputs from model inference
            ratio: Scale ratio from letterbox resize
            pad: Padding from letterbox resize
            
        Returns:
            Dictionary containing processed results with bounding boxes
        """
        # Get output
        outputs = np.transpose(np.squeeze(model_outputs[0]))
        rows = outputs.shape[0]
        
        boxes, scores, class_ids = [], [], []
        
        # Calculate gain factor
        gain = min(self.input_shape[2] / self.img_height, self.input_shape[3] / self.img_width)
        
        # Adjust for padding
        outputs[:, 0] -= pad[1]  # x
        outputs[:, 1] -= pad[0]  # y
        
        # Process detections
        for i in range(rows):
            classes_scores = outputs[i][4:]  # class probabilities
            max_score = np.amax(classes_scores)
            
            if max_score >= self.config["confidence_threshold"]:
                class_id = np.argmax(classes_scores)
                
                # Get bounding box coordinates
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                
                # Scale back to original image size
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)
                
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
        
        # Use common postprocess
        detections = self.common_postprocess(boxes, scores, class_ids, gain)
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "num_outputs": len(detections),
            "model_type": "yolo_openvino"
        }
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run complete YOLO prediction pipeline using OpenVINO
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess input
        preprocessed_data, ratio, pad = self.preprocess(input_data)
        
        # Run inference
        model_outputs = self._run_inference(preprocessed_data)
        
        # Postprocess outputs
        results = self.postprocess(model_outputs, ratio, pad)
        
        return results