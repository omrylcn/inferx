"""Comprehensive tests for settings system"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inferx.settings import get_inferx_settings


class TestSettingsBasics:
    """Test basic settings functionality"""
    
    def test_settings_loading(self):
        """Test settings can be loaded"""
        settings = get_inferx_settings()
        assert settings is not None
    
    def test_yolo_defaults(self):
        """Test YOLO default values"""
        settings = get_inferx_settings()
        
        # Input size validation
        assert settings.yolo_input_size > 0
        assert settings.yolo_input_size % 32 == 0
        assert settings.yolo_input_size <= 2048
        
        # Threshold validation
        assert 0.0 < settings.yolo_confidence_threshold < 1.0
        assert 0.0 < settings.yolo_nms_threshold < 1.0
        
        # Class names
        assert isinstance(settings.yolo_class_names, list)
        assert len(settings.yolo_class_names) > 0
        assert all(isinstance(name, str) for name in settings.yolo_class_names)
        
        # Max detections
        assert settings.yolo_max_detections > 0
        assert settings.yolo_max_detections <= 10000
    
    def test_anomaly_defaults(self):
        """Test anomaly detection defaults"""
        settings = get_inferx_settings()
        
        # Input size
        assert len(settings.anomaly_input_size) == 2
        assert all(size > 0 for size in settings.anomaly_input_size)
        
        # Threshold
        assert 0.0 <= settings.anomaly_threshold <= 1.0
        
        # Normalization
        assert isinstance(settings.anomaly_normalize, bool)
        assert len(settings.anomaly_mean) == 3
        assert len(settings.anomaly_std) == 3
    
    def test_classification_defaults(self):
        """Test classification defaults"""
        settings = get_inferx_settings()
        
        assert len(settings.classification_input_size) == 2
        assert settings.classification_top_k > 0
        assert settings.classification_top_k <= 10


class TestModelDetection:
    """Test model type detection"""
    
    def test_yolo_detection(self):
        """Test YOLO model detection"""
        settings = get_inferx_settings()
        
        yolo_models = [
            ("yolov8.onnx", "yolo_onnx"),
            ("yolo11n.onnx", "yolo_onnx"),
            ("ultralytics_yolo.onnx", "yolo_onnx"),
            ("yolov8.xml", "yolo_openvino"),
            ("yolo11n.xml", "yolo_openvino"),
        ]
        
        for model_name, expected_type in yolo_models:
            detected = settings.detect_model_type(Path(model_name))
            assert expected_type == detected, f"{model_name} should be {expected_type}, got {detected}"
    
    def test_anomaly_detection(self):
        """Test anomaly model detection"""
        settings = get_inferx_settings()
        
        anomaly_models = [
            ("padim_model.onnx", "anomalib"),
            ("patchcore_detector.xml", "anomalib"),
            ("anomaly_detection.onnx", "anomalib"),
            ("anomalib_model.xml", "anomalib"),
        ]
        
        for model_name, expected_type in anomaly_models:
            detected = settings.detect_model_type(Path(model_name))
            assert expected_type == detected, f"{model_name} should be {expected_type}, got {detected}"
    
    def test_generic_detection(self):
        """Test generic model detection"""
        settings = get_inferx_settings()
        
        generic_models = [
            ("model.onnx", "onnx"),
            ("classifier.onnx", "onnx"),
            ("model.xml", "openvino"),
            ("detector.xml", "openvino"),
        ]
        
        for model_name, expected_type in generic_models:
            detected = settings.detect_model_type(Path(model_name))
            assert expected_type == detected, f"{model_name} should be {expected_type}, got {detected}"


class TestModelDefaults:
    """Test model defaults retrieval"""
    
    def test_yolo_defaults_retrieval(self):
        """Test YOLO defaults retrieval"""
        settings = get_inferx_settings()
        
        yolo_defaults = settings.get_model_defaults("yolo")
        required_keys = ["input_size", "confidence_threshold", "nms_threshold", "max_detections", "class_names"]
        
        for key in required_keys:
            assert key in yolo_defaults, f"Missing key: {key}"
        
        # Test values match settings
        assert yolo_defaults["input_size"] == settings.yolo_input_size
        assert yolo_defaults["confidence_threshold"] == settings.yolo_confidence_threshold
    
    def test_anomaly_defaults_retrieval(self):
        """Test anomaly defaults retrieval"""
        settings = get_inferx_settings()
        
        anomaly_defaults = settings.get_model_defaults("anomaly")
        required_keys = ["input_size", "threshold", "normalize", "mean", "std", "return_anomaly_map"]
        
        for key in required_keys:
            assert key in anomaly_defaults, f"Missing key: {key}"
        
        # Test values match settings
        assert anomaly_defaults["input_size"] == settings.anomaly_input_size
        assert anomaly_defaults["threshold"] == settings.anomaly_threshold
    
    def test_unknown_model_defaults(self):
        """Test unknown model type returns empty dict"""
        settings = get_inferx_settings()
        
        unknown_defaults = settings.get_model_defaults("unknown_model_type")
        assert unknown_defaults == {}


class TestDeviceMapping:
    """Test device name mapping"""
    
    def test_basic_device_mapping(self):
        """Test basic device mapping"""
        settings = get_inferx_settings()
        
        # Test known devices
        assert settings.get_device_name("cpu") == "CPU"
        assert settings.get_device_name("auto") in ["AUTO", "CPU", "GPU"]  # Can be mapped to different values
        
        # Test case insensitivity
        assert settings.get_device_name("CPU") == "CPU"
        assert settings.get_device_name("Cpu") == "CPU"
    
    def test_unknown_device_mapping(self):
        """Test unknown device mapping"""
        settings = get_inferx_settings()
        
        # Unknown devices should return uppercase
        assert settings.get_device_name("unknown_device") == "UNKNOWN_DEVICE"
        assert settings.get_device_name("custom123") == "CUSTOM123"


class TestSettingsProperties:
    """Test settings property methods"""
    
    def test_yolo_defaults_property(self):
        """Test yolo_defaults property"""
        settings = get_inferx_settings()
        defaults = settings.yolo_defaults
        
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        
        # Check specific values
        assert defaults["input_size"] == settings.yolo_input_size
        assert defaults["confidence_threshold"] == settings.yolo_confidence_threshold
        assert defaults["class_names"] == settings.yolo_class_names
    
    def test_anomaly_defaults_property(self):
        """Test anomaly_defaults property"""
        settings = get_inferx_settings()
        defaults = settings.anomaly_defaults
        
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        
        # Check specific values
        assert defaults["input_size"] == settings.anomaly_input_size
        assert defaults["threshold"] == settings.anomaly_threshold
        assert defaults["normalize"] == settings.anomaly_normalize
    
    def test_classification_defaults_property(self):
        """Test classification_defaults property"""
        settings = get_inferx_settings()
        defaults = settings.classification_defaults
        
        assert isinstance(defaults, dict)
        assert len(defaults) > 0


# Integration test
def test_settings_integration():
    """Test settings work in realistic context"""
    settings = get_inferx_settings()
    
    # Test complete workflow
    model_path = Path("yolo_test.onnx")
    model_type = settings.detect_model_type(model_path)
    model_defaults = settings.get_model_defaults(model_type)
    device_name = settings.get_device_name("auto")
    
    assert model_type == "yolo_onnx"
    assert len(model_defaults) > 0
    assert device_name in ["AUTO", "CPU", "GPU"]


if __name__ == "__main__":
    import traceback
    
    test_classes = [TestSettingsBasics, TestModelDetection, TestModelDefaults, TestDeviceMapping, TestSettingsProperties]
    
    print("🧪 Settings Tests")
    print("=" * 30)
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                traceback.print_exc()
    
    # Integration test
    total_tests += 1
    try:
        test_settings_integration()
        print(f"\n✅ Integration test")
        passed_tests += 1
    except Exception as e:
        print(f"\n❌ Integration test: {e}")
        traceback.print_exc()
    
    print(f"\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All settings tests passed!")
    else:
        print("⚠️  Some settings tests failed")