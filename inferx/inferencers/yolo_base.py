"""Base YOLO inferencer with common functionality for ONNX and OpenVINO"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import logging

from .base import BaseInferencer
from ..utils import ImageProcessor
from ..exceptions import (
    ModelLoadError,
    InferenceFailedError,
    InputInvalidFormatError,
    ErrorCode
)

logger = logging.getLogger(__name__)


class BaseYOLOInferencer(BaseInferencer):
    """Base class for YOLO inferencers with common functionality"""
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict] = None):
        """Initialize base YOLO inferencer
        
        Args:
            model_path: Path to model file
            config: Optional configuration dictionary
        """
        # Default YOLO config
        default_config = {
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
        
        if config:
            default_config.update(config)
            
        super().__init__(model_path, default_config)
    
    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Letterbox resize image while maintaining aspect ratio
        
        Args:
            img: Input image
            new_shape: Target shape (height, width)
            
        Returns:
            Tuple of (resized_image, scale_ratio, padding)
        """
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # width, height
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, (r, r), (dw, dh)
    
    def common_preprocess(self, input_data: Any) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Common YOLO preprocessing: letterbox resize + normalize + transpose
        
        Args:
            input_data: Input data (image path or numpy array)
            
        Returns:
            Tuple of (preprocessed_image, scale_ratio, padding)
        """
        if isinstance(input_data, (str, Path)):
            # Load image from file
            img = ImageProcessor.load_image(input_data)
        elif isinstance(input_data, np.ndarray):
            img = input_data
        else:
            raise InputInvalidFormatError(
                input_path=str(input_data) if hasattr(input_data, '__str__') else "unknown",
                expected_formats=["image_path", "numpy_array"],
                context={"actual_type": type(input_data).__name__}
            )
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get input size from config
        input_size = self.config.get("input_size", 640)
        
        # Letterbox resize
        img, ratio, pad = self._letterbox(img, (input_size, input_size))
        
        # Normalize to [0, 1]
        img = np.array(img) / 255.0
        
        # Transpose HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img, ratio, pad
    
    def common_postprocess(self, boxes: List, scores: List, class_ids: List, gain: float) -> List[Dict]:
        """Common YOLO postprocessing: NMS + format results
        
        Args:
            boxes: List of bounding boxes
            scores: List of confidence scores
            class_ids: List of class IDs
            gain: Scale gain factor
            
        Returns:
            List of detection dictionaries
        """
        # Apply Non-Maximum Suppression using OpenCV
        detections = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                scores, 
                self.config["confidence_threshold"], 
                self.config["nms_threshold"]
            )
            
            # Format final results
            if len(indices) > 0:
                # Handle different OpenCV NMS return formats
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                for i in indices:
                    if isinstance(i, (list, np.ndarray)):
                        i = i[0]
                    
                    box = boxes[i]
                    score = scores[i]
                    class_id = class_ids[i]
                    
                    detection = {
                        "bbox": [float(x) for x in box],  # [x, y, width, height]
                        "confidence": float(score),
                        "class_id": int(class_id),
                        "class_name": self.config["class_names"][class_id] if class_id < len(self.config["class_names"]) else f"class_{class_id}"
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image
        
        Args:
            img: Input image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        # Generate a color palette for the classes
        class_names = self.config.get("class_names", [])
        color_palette = np.random.uniform(0, 255, size=(len(class_names), 3))
        
        img_copy = img.copy()
        
        for detection in detections:
            # Extract the coordinates of the bounding box
            bbox = detection["bbox"]
            x, y, w, h = bbox
            
            # Get class ID and confidence
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            
            # Retrieve the color for the class ID
            color = color_palette[class_id]
            
            # Draw the bounding box on the image
            cv2.rectangle(img_copy, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Create the label text with class name and score
            class_name = detection.get("class_name", f"class_{class_id}")
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Calculate the position of the label text
            label_x = int(x)
            label_y = int(y) - 10 if int(y) - 10 > label_height else int(y) + 10
            
            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img_copy, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
            
            # Draw the label text on the image
            cv2.putText(img_copy, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return img_copy
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the YOLO model
        
        Returns:
            Dictionary containing model metadata
        """
        base_info = super().get_model_info()
        
        # Add YOLO-specific information
        yolo_info = {
            "model_type": "yolo",
            "input_size": self.config.get("input_size", 640),
            "confidence_threshold": self.config["confidence_threshold"],
            "nms_threshold": self.config["nms_threshold"],
            "num_classes": len(self.config["class_names"])
        }
        
        base_info.update(yolo_info)
        return base_info