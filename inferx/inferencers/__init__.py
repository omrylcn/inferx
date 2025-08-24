"""Inferencer modules for different model types"""

from .base import BaseInferencer
from .onnx_inferencer import ONNXInferencer
from .openvino_inferencer import OpenVINOInferencer
from .yolo_base import BaseYOLOInferencer
from .yolo import YOLOInferencer
from .yolo_openvino import YOLOOpenVINOInferencer


inferencer_dict = {
    "onnx": ONNXInferencer,
    "openvino": OpenVINOInferencer,
    "yolo": YOLOInferencer,
    "yolo_openvino": YOLOOpenVINOInferencer
}
__all__ = [
    "BaseInferencer", 
    "ONNXInferencer", 
    "OpenVINOInferencer",
    "YOLOInferencer", 
    "YOLOOpenVINOInferencer"
]