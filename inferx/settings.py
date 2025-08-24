"""
InferX Configuration with Pydantic Settings

Clean, direct implementation using pydantic-settings with YAML support
Replaces the complex config.py system with type-safe, validated configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from pathlib import Path
import yaml
from functools import lru_cache

# =============================================================================
# ENUMS
# =============================================================================

class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    MYRIAD = "myriad"
    HDDL = "hddl"

class RuntimeType(str, Enum):
    AUTO = "auto"
    ONNX = "onnx"
    OPENVINO = "openvino"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# CORE INFERX SETTINGS
# =============================================================================

class InferXSettings(BaseSettings):
    """
    Main InferX Configuration
    
    Loads from:
    1. Environment variables (INFERX_*)
    2. .env file
    3. YAML config files (hierarchy)
    4. Default values
    """
    
    
    # =========================================================================
    # MODEL DETECTION
    # =========================================================================
    yolo_keywords: List[str] = Field(
        default=["yolo", "yolov", "yolov8", "yolov11", "ultralytics"],
        description="Keywords for YOLO model detection"
    )
    classification_keywords: List[str] = Field(
        default=["resnet", "efficientnet", "mobilenet", "classifier"],
        description="Keywords for classification model detection"
    )
    anomalib_keywords: List[str] = Field(
        default=["anomalib", "padim", "patchcore", "anomaly"],
        description="Keywords for anomaly detection"
    )
    
    # =========================================================================
    # DEVICE MAPPING
    # =========================================================================
    device_auto: str = Field(default="AUTO", alias="device_mapping_auto")
    device_cpu: str = Field(default="CPU", alias="device_mapping_cpu")
    device_gpu: str = Field(default="GPU", alias="device_mapping_gpu")
    device_myriad: str = Field(default="MYRIAD", alias="device_mapping_myriad")
    device_hddl: str = Field(default="HDDL", alias="device_mapping_hddl")
    
    @property
    def device_mapping(self) -> Dict[str, str]:
        """Get device mapping dictionary"""
        return {
            "auto": self.device_auto,
            "cpu": self.device_cpu,
            "gpu": self.device_gpu,
            "myriad": self.device_myriad,
            "hddl": self.device_hddl
        }
    
    # =========================================================================
    # MODEL DEFAULTS - YOLO
    # =========================================================================
    yolo_input_size: int = Field(
        default=640,
        gt=0,
        description="YOLO input size"
    )
    yolo_confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="YOLO confidence threshold"
    )
    yolo_nms_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="YOLO NMS threshold"
    )
    yolo_max_detections: int = Field(
        default=100,
        gt=0,
        le=10000,
        description="YOLO max detections"
    )
    yolo_class_names: List[str] = Field(
        default_factory=lambda: [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light"
        ],
        description="YOLO class names"
    )
    
    @validator('yolo_input_size')
    def validate_yolo_input_size(cls, v):
        if v % 32 != 0:
            raise ValueError('YOLO input size must be divisible by 32')
        return v
    
    @property
    def yolo_defaults(self) -> Dict[str, Any]:
        """Get YOLO model defaults"""
        return {
            "input_size": self.yolo_input_size,
            "confidence_threshold": self.yolo_confidence_threshold,
            "nms_threshold": self.yolo_nms_threshold,
            "max_detections": self.yolo_max_detections,
            "class_names": self.yolo_class_names
        }
    
    # =========================================================================
    # MODEL DEFAULTS - CLASSIFICATION
    # =========================================================================
    classification_input_size: List[int] = Field(
        default=[224, 224],
        description="Classification input size [H, W]"
    )
    classification_top_k: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Classification top-k"
    )
    classification_normalize_mean: List[float] = Field(
        default=[0.485, 0.456, 0.406],
        description="ImageNet normalization mean"
    )
    classification_normalize_std: List[float] = Field(
        default=[0.229, 0.224, 0.225],
        description="ImageNet normalization std"
    )
    
    @property
    def classification_defaults(self) -> Dict[str, Any]:
        """Get classification model defaults"""
        return {
            "input_size": self.classification_input_size,
            "top_k": self.classification_top_k,
            "normalize": {
                "mean": self.classification_normalize_mean,
                "std": self.classification_normalize_std
            }
        }
    
    # =========================================================================
    # PREPROCESSING DEFAULTS
    # =========================================================================
    onnx_target_size: List[int] = Field(default=[224, 224])
    onnx_normalize: bool = Field(default=True)
    onnx_color_format: Literal["RGB", "BGR"] = Field(default="RGB")
    onnx_maintain_aspect_ratio: bool = Field(default=True)
    onnx_normalize_mean: List[float] = Field(default=[0.485, 0.456, 0.406])
    onnx_normalize_std: List[float] = Field(default=[0.229, 0.224, 0.225])
    
    @property
    def onnx_preprocessing_defaults(self) -> Dict[str, Any]:
        """Get ONNX preprocessing defaults"""
        return {
            "target_size": self.onnx_target_size,
            "normalize": self.onnx_normalize,
            "color_format": self.onnx_color_format,
            "maintain_aspect_ratio": self.onnx_maintain_aspect_ratio,
            "mean": self.onnx_normalize_mean,
            "std": self.onnx_normalize_std
        }
    
    # =========================================================================
    # RUNTIME PREFERENCES
    # =========================================================================
    onnx_gpu_providers: List[str] = Field(
        default=["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
    )
    onnx_cpu_providers: List[str] = Field(
        default=["CPUExecutionProvider"]
    )
    onnx_graph_optimization: Literal[
        "ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"
    ] = Field(default="ORT_ENABLE_ALL")
    onnx_inter_op_threads: int = Field(default=0, ge=0)
    onnx_intra_op_threads: int = Field(default=0, ge=0)
    
    @property
    def onnx_runtime_preferences(self) -> Dict[str, Any]:
        """Get ONNX runtime preferences"""
        return {
            "providers": {
                "gpu": self.onnx_gpu_providers,
                "cpu": self.onnx_cpu_providers,
                "auto": self.onnx_gpu_providers
            },
            "session_options": {
                "graph_optimization_level": self.onnx_graph_optimization,
                "inter_op_num_threads": self.onnx_inter_op_threads,
                "intra_op_num_threads": self.onnx_intra_op_threads
            }
        }
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_model_loading: bool = Field(default=True)
    log_inference_timing: bool = Field(default=True)
    log_preprocessing: bool = Field(default=False)
    log_device_info: bool = Field(default=True)
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": self.log_level.value,
            "format": self.log_format,
            "categories": {
                "model_loading": self.log_model_loading,
                "inference_timing": self.log_inference_timing,
                "preprocessing": self.log_preprocessing,
                "device_info": self.log_device_info
            }
        }
    
    # =========================================================================
    # METHODS FOR COMPATIBILITY
    # =========================================================================
    
    def get_model_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get model defaults for compatibility with existing code"""
        base_type = model_type.split('_')[0].lower()
        
        if base_type == "yolo":
            return self.yolo_defaults
        elif base_type == "classification":
            return self.classification_defaults
        else:
            return {}
    
    def get_preprocessing_defaults(self, runtime: str) -> Dict[str, Any]:
        """Get preprocessing defaults for compatibility"""
        if runtime.lower() == "onnx":
            return self.onnx_preprocessing_defaults
        # Add openvino when needed
        return {}
    
    def get_device_name(self, device: str) -> str:
        """Map device name for compatibility"""
        # Try to get from flattened device mapping fields
        device_key = f"device_{device.lower()}"
        if hasattr(self, device_key):
            return getattr(self, device_key)
        # Fallback to uppercase
        return device.upper()
    
    def detect_model_type(self, model_path: Path, runtime_hint: Optional[str] = None) -> str:
        """Detect model type from filename"""
        filename = model_path.name.lower()
        
        # Check YOLO keywords
        if any(keyword in filename for keyword in self.yolo_keywords):
            if model_path.suffix.lower() == ".xml":
                return "yolo_openvino"
            return "yolo_onnx"
        
        # Check classification keywords
        if any(keyword in filename for keyword in self.classification_keywords):
            return "classification"
        
        # Check anomaly keywords
        if any(keyword in filename for keyword in self.anomalib_keywords):
            return "anomalib"
        
        # Default based on file extension
        if model_path.suffix.lower() == ".xml":
            return "openvino"
        elif model_path.suffix.lower() == ".onnx":
            return "onnx"
        else:
            return "generic"
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    model_config = SettingsConfigDict(
        env_prefix="INFERX_",
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from default.yaml
    )


# =============================================================================
# TEMPLATE PROJECT SETTINGS
# =============================================================================

class YoloTemplateConfigSettings(BaseSettings):
    """
    YOLO Template Project Configuration Settings
    
    Standalone class for validating YOLO template config.yaml files
    Uses pydantic-settings directly - independent of InferXSettings
    CLI loads config.yaml and validates with this class
    """
    
    # Model section
    model_path: str = Field(description="Path to YOLO model file")
    model_type: str = Field(default="yolo", description="Model type")
    
    # Inference section  
    device: DeviceType = Field(default=DeviceType.AUTO, description="Inference device")
    runtime: RuntimeType = Field(default=RuntimeType.AUTO, description="Runtime engine")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="YOLO confidence threshold")
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="YOLO NMS threshold")
    input_size: int = Field(default=640, gt=0, description="YOLO input size")
    batch_size: Optional[int] = Field(default=1, gt=0, le=64, description="Batch size")
    
    # Preprocessing section
    target_size: List[int] = Field(default=[640, 640], description="Preprocessing target size")
    normalize: bool = Field(default=True, description="Apply normalization")
    color_format: Literal["RGB", "BGR"] = Field(default="RGB", description="Color format")
    maintain_aspect_ratio: bool = Field(default=True, description="Maintain aspect ratio")
    
    @validator('input_size')
    def validate_yolo_input_size(cls, v):
        if v % 32 != 0:
            raise ValueError('YOLO input size must be divisible by 32')
        return v
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not v or not v.strip():
            raise ValueError('Model path cannot be empty')
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="YOLO_TEMPLATE_",
        case_sensitive=False
    )
    
    # Inference section  
    device: DeviceType = Field(default=DeviceType.AUTO)
    runtime: RuntimeType = Field(default=RuntimeType.AUTO)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_threshold: Optional[float] = Field(default=0.45, ge=0.0, le=1.0)
    input_size: int = Field(default=640, gt=0)
    batch_size: Optional[int] = Field(default=1, gt=0, le=64)
    
    # Preprocessing section
    target_size: List[int] = Field(default=[640, 640])
    normalize: bool = Field(default=True)
    color_format: Literal["RGB", "BGR"] = Field(default="RGB")
    maintain_aspect_ratio: bool = Field(default=True)
    
    @validator('input_size')
    def validate_input_size(cls, v, values):
        model_type = values.get('model_type', '')
        if model_type.startswith('yolo') and v % 32 != 0:
            raise ValueError('YOLO input size must be divisible by 32')
        return v
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not v or not v.strip():
            raise ValueError('Model path cannot be empty')
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="TEMPLATE_",
        case_sensitive=False
    )


# =============================================================================
# CONFIG LOADING WITH HIERARCHY
# =============================================================================

class ConfigHierarchy:
    """Handle InferX config file hierarchy loading"""
    
    @staticmethod
    def get_config_paths() -> List[Path]:
        """Get config file paths in priority order (highest to lowest)"""
        return [
            Path.cwd() / "inferx_config.yaml",  # Project local
            Path.home() / ".inferx" / "config.yaml",  # User global
            Path(__file__).parent / "configs" / "default.yaml"  # System default
        ]
    
    @staticmethod
    def load_merged_config() -> Dict[str, Any]:
        """Load and merge configs from hierarchy"""
        merged_config = {}
        
        # Load in reverse order (lowest to highest priority)
        for config_path in reversed(ConfigHierarchy.get_config_paths()):
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f) or {}
                    
                    # Merge configs
                    ConfigHierarchy._deep_merge(merged_config, file_config)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {config_path}: {e}")
        
        return merged_config
    
    @staticmethod
    def _deep_merge(target: Dict, source: Dict) -> None:
        """Recursively merge source into target"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigHierarchy._deep_merge(target[key], value)
            else:
                target[key] = value


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

@lru_cache()
def get_inferx_settings() -> InferXSettings:
    """Get InferX settings with hierarchy loading and caching"""
    
    # Load merged config from hierarchy
    merged_config = ConfigHierarchy.load_merged_config()
    
    # Convert nested dict to flat structure for Pydantic
    flat_config = _flatten_config(merged_config)
    
    # Create settings with merged config
    return InferXSettings(**flat_config)


def validate_yolo_template_config(config_path: Path) -> YoloTemplateConfigSettings:
    """Validate YOLO template project configuration with pydantic-settings"""
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Flatten nested YAML structure for pydantic
        flat_data = {}
        
        # Extract model section
        if 'model' in config_data:
            flat_data['model_path'] = config_data['model']['path']
            flat_data['model_type'] = config_data['model']['type']
        
        # Extract inference section
        if 'inference' in config_data:
            inference = config_data['inference']
            flat_data.update({
                'device': inference.get('device', 'auto'),
                'runtime': inference.get('runtime', 'auto'),
                'confidence_threshold': inference.get('confidence_threshold', 0.25),
                'nms_threshold': inference.get('nms_threshold', 0.45),
                'input_size': inference.get('input_size', 640),
                'batch_size': inference.get('batch_size', 1)
            })
        
        # Extract preprocessing section
        if 'preprocessing' in config_data:
            preprocessing = config_data['preprocessing']
            flat_data.update({
                'target_size': preprocessing.get('target_size', [640, 640]),
                'normalize': preprocessing.get('normalize', True),
                'color_format': preprocessing.get('color_format', 'RGB'),
                'maintain_aspect_ratio': preprocessing.get('maintain_aspect_ratio', True)
            })
        
        # Create and validate with pydantic
        return YoloTemplateConfigSettings(**flat_data)
    
    except Exception as e:
        raise ValueError(f"Failed to validate YOLO template config {config_path}: {e}")


def validate_template_config(config_path: Path):
    """Generic template config validation - defaults to YOLO"""
    return validate_yolo_template_config(config_path)


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config for Pydantic"""
    flat = {}
    
    for key, value in config.items():
        flat_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Handle special nested structures
            if key == "model_defaults":
                for model_type, model_config in value.items():
                    for setting_key, setting_value in model_config.items():
                        flat[f"{model_type}_{setting_key}"] = setting_value
            elif key == "device_mapping":
                for device, mapping in value.items():
                    flat[f"device_{device}"] = mapping
            elif key == "preprocessing_defaults":
                for runtime, runtime_config in value.items():
                    for setting_key, setting_value in runtime_config.items():
                        flat[f"{runtime}_{setting_key}"] = setting_value
            elif key == "logging":
                flat["log_level"] = value.get("level", "INFO")
                flat["log_format"] = value.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                categories = value.get("categories", {})
                for cat_key, cat_value in categories.items():
                    flat[f"log_{cat_key}"] = cat_value
            else:
                # Generic flattening
                flat.update(_flatten_config(value, f"{flat_key}_"))
        else:
            flat[flat_key] = value
    
    return flat


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global settings instance with caching
settings = get_inferx_settings()


# =============================================================================
# COMPATIBILITY FUNCTIONS
# =============================================================================

def get_config(config_path: Optional[Path] = None) -> InferXSettings:
    """Get configuration - compatibility function"""
    # For now, ignore config_path and use hierarchy
    # Can be enhanced later if needed
    return settings


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("InferX Settings with Pydantic")
    print("=" * 40)
    
    # Load settings
    config = get_inferx_settings()
    
    print(f"YOLO input size: {config.yolo_input_size}")
    print(f"YOLO confidence: {config.yolo_confidence_threshold}")
    print(f"Device mapping: {config.device_mapping}")
    print(f"Log level: {config.log_level}")
    
    # Test model detection
    test_path = Path("yolov8_cars.onnx")
    detected_type = config.detect_model_type(test_path)
    print(f"Detected model type for '{test_path}': {detected_type}")
    
    # Test model defaults
    yolo_defaults = config.get_model_defaults("yolo")
    print(f"YOLO defaults: {len(yolo_defaults)} settings")
    
    # Test compatibility methods
    device_name = config.get_device_name("gpu")
    print(f"Device 'gpu' maps to: {device_name}")
    
    print("\n✅ InferX Settings loaded successfully!")
    print("🎯 Benefits:")
    print("   • Type-safe configuration")
    print("   • Environment variable support")
    print("   • YAML hierarchy loading") 
    print("   • Automatic validation")
    print("   • Clean, maintainable code")