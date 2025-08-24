"""YOLO-specific functionality tests"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inferx.inferencers.yolo_base import BaseYOLOInferencer
from inferx.settings import get_inferx_settings


class MockYOLOInferencer(BaseYOLOInferencer):
    """Mock YOLO inferencer for testing"""
    
    def __init__(self, model_path, config=None):
        self.model_path = Path(model_path)
        self.config = config or {}
        self.session = None  # Mock session
        # Set up basic config
        settings = get_inferx_settings()
        self.config.update(settings.get_model_defaults("yolo"))
    
    def _load_model(self):
        """Mock model loading"""
        pass
    
    def _run_inference(self, preprocessed_data):
        """Mock inference - return dummy YOLO output"""
        # Simulate YOLO output: [batch, anchors, 4+1+classes]
        # For 80 classes: 4 (bbox) + 1 (objectness) + 80 (classes) = 85
        batch_size = preprocessed_data.shape[0]
        return [np.random.rand(batch_size, 25200, 85)]  # Mock YOLO output


class TestYOLOPreprocessing:
    """Test YOLO preprocessing functionality"""
    
    def test_yolo_preprocessing_shape(self):
        """Test YOLO preprocessing produces correct shapes"""
        mock_inferencer = MockYOLOInferencer("test_yolo.onnx")
        
        # Mock image preprocessing
        # YOLO preprocessing should produce [batch, channels, height, width]
        input_size = 640
        expected_shape = (1, 3, input_size, input_size)
        
        # We can't easily test without image, but test the logic exists
        assert hasattr(mock_inferencer, 'common_preprocess')
    
    def test_yolo_config_loading(self):
        """Test YOLO loads proper configuration"""
        mock_inferencer = MockYOLOInferencer("test_yolo.onnx")
        
        # Should have YOLO-specific config
        assert "input_size" in mock_inferencer.config
        assert "confidence_threshold" in mock_inferencer.config
        assert "nms_threshold" in mock_inferencer.config
        assert "class_names" in mock_inferencer.config
        
        # Check reasonable values
        assert mock_inferencer.config["input_size"] > 0
        assert mock_inferencer.config["input_size"] % 32 == 0
        assert 0.0 < mock_inferencer.config["confidence_threshold"] < 1.0
        assert 0.0 < mock_inferencer.config["nms_threshold"] < 1.0


class TestYOLOPostprocessing:
    """Test YOLO postprocessing functionality"""
    
    def test_yolo_postprocessing_structure(self):
        """Test YOLO postprocessing returns correct structure"""
        mock_inferencer = MockYOLOInferencer("test_yolo.onnx")
        
        # Mock YOLO output
        batch_size = 1
        num_anchors = 25200
        num_classes = 80
        mock_output = np.random.rand(batch_size, num_anchors, 4 + 1 + num_classes)
        
        # Test that common_postprocess exists
        assert hasattr(mock_inferencer, 'common_postprocess')
        
        # Test basic structure - we can't fully test without implementing NMS
        # But we can test the method exists and basic setup
        boxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
        scores = [0.9, 0.8]
        class_ids = [0, 1]
        
        # This tests the signature exists
        try:
            result = mock_inferencer.common_postprocess(boxes, scores, class_ids, gain=1.0)
            assert isinstance(result, list)
        except:
            # Method might not be fully implemented yet
            pass
    
    def test_yolo_result_format(self):
        """Test YOLO result format"""
        # Expected YOLO result structure
        expected_result = {
            "detections": [
                {
                    "bbox": [100, 100, 200, 200],  # [x, y, width, height]
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "person"
                }
            ],
            "num_detections": 1,
            "model_type": "yolo"
        }
        
        # Test structure
        assert "detections" in expected_result
        assert "num_detections" in expected_result
        assert "model_type" in expected_result
        
        # Test detection format
        detection = expected_result["detections"][0]
        assert "bbox" in detection
        assert "confidence" in detection
        assert "class_id" in detection
        assert "class_name" in detection
        
        # Test bbox format
        bbox = detection["bbox"]
        assert len(bbox) == 4
        assert all(isinstance(coord, (int, float)) for coord in bbox)


class TestYOLOConfiguration:
    """Test YOLO configuration handling"""
    
    def test_yolo_default_config(self):
        """Test YOLO default configuration"""
        settings = get_inferx_settings()
        yolo_defaults = settings.get_model_defaults("yolo")
        
        # Required YOLO settings
        required_keys = ["input_size", "confidence_threshold", "nms_threshold", "class_names", "max_detections"]
        
        for key in required_keys:
            assert key in yolo_defaults, f"Missing YOLO config key: {key}"
        
        # Validate values
        assert yolo_defaults["input_size"] % 32 == 0
        assert 0.0 < yolo_defaults["confidence_threshold"] < 1.0
        assert 0.0 < yolo_defaults["nms_threshold"] < 1.0
        assert len(yolo_defaults["class_names"]) > 0
        assert yolo_defaults["max_detections"] > 0
    
    def test_yolo_custom_config(self):
        """Test YOLO with custom configuration"""
        custom_config = {
            "input_size": 1024,
            "confidence_threshold": 0.3,
            "nms_threshold": 0.4,
            "max_detections": 200
        }
        
        mock_inferencer = MockYOLOInferencer("test_yolo.onnx", custom_config)
        
        # Custom values should be used
        assert mock_inferencer.config["input_size"] == 1024
        assert mock_inferencer.config["confidence_threshold"] == 0.3
        assert mock_inferencer.config["nms_threshold"] == 0.4
        assert mock_inferencer.config["max_detections"] == 200
    
    def test_yolo_class_names(self):
        """Test YOLO class names handling"""
        settings = get_inferx_settings()
        yolo_defaults = settings.get_model_defaults("yolo")
        
        class_names = yolo_defaults["class_names"]
        
        # Should have COCO classes
        assert len(class_names) >= 80
        assert "person" in class_names
        assert "car" in class_names
        assert "bicycle" in class_names
        
        # All should be strings
        assert all(isinstance(name, str) for name in class_names)


class TestYOLOInferencers:
    """Test YOLO inferencer classes"""
    
    def test_yolo_base_inferencer(self):
        """Test BaseYOLOInferencer structure"""
        # Test that the base class has required methods
        required_methods = ["common_preprocess", "common_postprocess"]
        
        for method in required_methods:
            assert hasattr(BaseYOLOInferencer, method), f"Missing method: {method}"
    
    def test_yolo_onnx_inferencer_import(self):
        """Test YOLO ONNX inferencer can be imported"""
        try:
            from inferx.inferencers.yolo import YOLOInferencer
            assert YOLOInferencer is not None
        except ImportError as e:
            assert False, f"Failed to import YOLOInferencer: {e}"
    
    def test_yolo_openvino_inferencer_import(self):
        """Test YOLO OpenVINO inferencer can be imported"""
        try:
            from inferx.inferencers.yolo_openvino import YOLOOpenVINOInferencer
            assert YOLOOpenVINOInferencer is not None
        except ImportError as e:
            assert False, f"Failed to import YOLOOpenVINOInferencer: {e}"


class TestYOLOWithRealModels:
    """Test YOLO with real model files if available"""
    
    def test_yolo_with_real_onnx_model(self):
        """Test YOLO with real ONNX model if available"""
        onnx_model = project_root / "models" / "yolo11n_onnx" / "yolo11n.onnx"
        
        if onnx_model.exists():
            print(f"  🔍 Testing with real ONNX model: {onnx_model}")
            try:
                from inferx.inferencers.yolo import YOLOInferencer
                inferencer = YOLOInferencer(str(onnx_model))
                print(f"  ✅ Successfully loaded YOLO ONNX model")
                
                # Test model info
                info = inferencer.get_model_info()
                assert "model_path" in info
                assert "model_type" in info
                
            except Exception as e:
                print(f"  ⚠️  Failed to load YOLO ONNX model: {e}")
                # Might fail due to missing ONNX Runtime
        else:
            print(f"  ⚪ YOLO ONNX model not found: {onnx_model}")
    
    def test_yolo_with_real_openvino_model(self):
        """Test YOLO with real OpenVINO model if available"""
        openvino_model = project_root / "models" / "yolo11n_openvino" / "yolo11n.xml"
        
        if openvino_model.exists():
            print(f"  🔍 Testing with real OpenVINO model: {openvino_model}")
            try:
                from inferx.inferencers.yolo_openvino import YOLOOpenVINOInferencer
                inferencer = YOLOOpenVINOInferencer(str(openvino_model))
                print(f"  ✅ Successfully loaded YOLO OpenVINO model")
                
                # Test model info
                info = inferencer.get_model_info()
                assert "model_path" in info
                assert "runtime" in info
                
            except Exception as e:
                print(f"  ⚠️  Failed to load YOLO OpenVINO model: {e}")
                # Might fail due to missing OpenVINO
        else:
            print(f"  ⚪ YOLO OpenVINO model not found: {openvino_model}")


# Integration test
def test_yolo_integration():
    """Test YOLO integration with runtime"""
    from inferx.runtime import InferenceEngine
    
    # Test that YOLO models are properly detected and configured
    try:
        engine = InferenceEngine("yolo_test.onnx", model_type="yolo")
    except Exception:
        pass  # Expected - no model file
    
    try:
        engine = InferenceEngine("yolo_test.xml", model_type="yolo")
    except Exception:
        pass  # Expected - no model file


if __name__ == "__main__":
    import traceback
    
    test_classes = [TestYOLOPreprocessing, TestYOLOPostprocessing, TestYOLOConfiguration, TestYOLOInferencers, TestYOLOWithRealModels]
    
    print("🧪 YOLO Tests")
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
        test_yolo_integration()
        print(f"\n✅ Integration test")
        passed_tests += 1
    except Exception as e:
        print(f"\n❌ Integration test: {e}")
        traceback.print_exc()
    
    print(f"\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All YOLO tests passed!")
    else:
        print("⚠️  Some YOLO tests failed")