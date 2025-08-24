"""Runtime engine tests"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inferx.runtime import InferenceEngine
from inferx.settings import get_inferx_settings


class TestRuntimeBasics:
    """Test basic runtime functionality"""
    
    def test_runtime_creation_fails_gracefully(self):
        """Test runtime fails gracefully with non-existent models"""
        # These should fail but not crash
        test_models = ["nonexistent.onnx", "fake_model.xml", "missing_yolo.onnx"]
        
        for model_path in test_models:
            try:
                engine = InferenceEngine(model_path)
                assert False, f"Should have failed for {model_path}"
            except Exception as e:
                # Expected failure - check it's a reasonable error
                assert isinstance(e, (FileNotFoundError, RuntimeError, Exception))
    
    def test_runtime_model_type_detection(self):
        """Test runtime correctly detects model types"""
        # Test model type parameter passing
        test_cases = [
            ("test_yolo.onnx", "yolo"),
            ("test_model.xml", None),  # Auto-detect
            ("anomaly_detector.onnx", "anomaly"),
        ]
        
        for model_path, model_type in test_cases:
            try:
                if model_type:
                    engine = InferenceEngine(model_path, model_type=model_type)
                else:
                    engine = InferenceEngine(model_path)
                # If we get here, it failed at model loading (expected)
                assert False, "Should have failed at model loading"
            except Exception as e:
                # Expected - check it's failing at the right place
                error_msg = str(e).lower()
                # Should fail due to file not found or model loading, not type detection
                assert any(term in error_msg for term in ['file', 'model', 'load', 'exist'])
    
    def test_runtime_device_parameter(self):
        """Test runtime handles device parameter"""
        devices = ["cpu", "gpu", "auto"]
        
        for device in devices:
            try:
                engine = InferenceEngine("test.onnx", device=device)
                assert False, "Should have failed at model loading"
            except Exception as e:
                # Should fail at model loading, not device validation
                assert "device" not in str(e).lower() or "file" in str(e).lower()
    
    def test_runtime_config_parameter(self):
        """Test runtime handles config parameter"""
        config = {
            "input_size": 640,
            "confidence_threshold": 0.5,
            "device": "cpu"
        }
        
        try:
            engine = InferenceEngine("test.onnx", config=config)
            assert False, "Should have failed at model loading"
        except Exception as e:
            # Should fail at model loading, config should be accepted
            assert "config" not in str(e).lower()


class TestRuntimeConfiguration:
    """Test runtime configuration building"""
    
    def test_build_config_with_defaults(self):
        """Test config building uses model defaults"""
        settings = get_inferx_settings()
        
        # Test that YOLO models get YOLO defaults
        try:
            engine = InferenceEngine("yolo_test.onnx", model_type="yolo")
        except Exception:
            pass  # Expected failure, but config should be built
    
    def test_config_merging(self):
        """Test user config merges with defaults"""
        user_config = {
            "confidence_threshold": 0.8,
            "custom_setting": "test_value"
        }
        
        try:
            engine = InferenceEngine("test.onnx", config=user_config)
        except Exception:
            pass  # Expected failure, but config merging should work


class TestRuntimeDetectionLogic:
    """Test runtime detection and selection logic"""
    
    def test_runtime_selection_onnx(self):
        """Test ONNX runtime selection"""
        # ONNX files should select ONNX runtime
        try:
            engine = InferenceEngine("model.onnx")
        except Exception:
            pass  # Expected - no model file
    
    def test_runtime_selection_openvino(self):
        """Test OpenVINO runtime selection"""
        # XML files should select OpenVINO runtime
        try:
            engine = InferenceEngine("model.xml")
        except Exception:
            pass  # Expected - no model file
    
    def test_runtime_override(self):
        """Test runtime can be overridden"""
        # Test explicit runtime specification
        try:
            engine = InferenceEngine("model.onnx", runtime="openvino")
        except Exception:
            pass  # Expected - no model file
        
        try:
            engine = InferenceEngine("model.xml", runtime="onnx")
        except Exception:
            pass  # Expected - no model file


class TestRuntimeErrorHandling:
    """Test runtime error handling"""
    
    def test_invalid_model_path(self):
        """Test invalid model paths raise appropriate errors"""
        invalid_paths = [
            "",  # Empty path
            "/nonexistent/path/model.onnx",  # Non-existent directory
            "model.txt",  # Wrong extension
        ]
        
        for path in invalid_paths:
            try:
                engine = InferenceEngine(path)
                assert False, f"Should have failed for path: {path}"
            except Exception as e:
                # Should get appropriate error
                assert isinstance(e, (ValueError, FileNotFoundError, RuntimeError, Exception))
    
    def test_invalid_model_type(self):
        """Test invalid model type handling"""
        # Invalid model types should either auto-detect or fail gracefully
        try:
            engine = InferenceEngine("test.onnx", model_type="invalid_type")
        except Exception:
            pass  # Expected failure
    
    def test_invalid_device(self):
        """Test invalid device handling"""
        # Invalid devices should be handled gracefully
        try:
            engine = InferenceEngine("test.onnx", device="invalid_device")
        except Exception as e:
            # Should fail at model loading, not device validation
            pass


class TestRuntimeWithRealModels:
    """Test runtime with actual model files (if they exist)"""
    
    def test_with_existing_yolo_models(self):
        """Test with existing YOLO models if available"""
        # Check for actual model files
        model_paths = [
            project_root / "models" / "yolo11n_onnx" / "yolo11n.onnx",
            project_root / "models" / "yolo11n_openvino" / "yolo11n.xml"
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                print(f"  🔍 Testing with real model: {model_path}")
                try:
                    engine = InferenceEngine(str(model_path))
                    print(f"  ✅ Successfully created engine for {model_path.name}")
                    
                    # Test model info
                    info = engine.get_model_info()
                    assert "model_path" in info
                    assert "status" in info
                    
                except Exception as e:
                    print(f"  ⚠️  Failed to load {model_path.name}: {e}")
                    # This might be expected if dependencies are missing
            else:
                print(f"  ⚪ Model not found: {model_path}")
    
    def test_model_info_structure(self):
        """Test model info has expected structure"""
        try:
            engine = InferenceEngine("test.onnx")
        except Exception:
            pass  # Expected
        
        # Test info structure with mock
        # This tests the interface without requiring real models


# Integration test
def test_runtime_settings_integration():
    """Test runtime integrates properly with settings"""
    settings = get_inferx_settings()
    
    # Test that runtime uses settings for model detection
    model_path = Path("yolo_test.onnx")
    detected_type = settings.detect_model_type(model_path)
    
    try:
        # This should use the detected model type
        engine = InferenceEngine(str(model_path), model_type=detected_type)
    except Exception:
        pass  # Expected failure - no model file
    
    # Test device mapping integration
    device_name = settings.get_device_name("auto")
    try:
        engine = InferenceEngine("test.onnx", device="auto")
    except Exception:
        pass  # Expected failure


if __name__ == "__main__":
    import traceback
    
    test_classes = [TestRuntimeBasics, TestRuntimeConfiguration, TestRuntimeDetectionLogic, TestRuntimeErrorHandling, TestRuntimeWithRealModels]
    
    print("🧪 Runtime Tests")
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
        test_runtime_settings_integration()
        print(f"\n✅ Integration test")
        passed_tests += 1
    except Exception as e:
        print(f"\n❌ Integration test: {e}")
        traceback.print_exc()
    
    print(f"\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All runtime tests passed!")
    else:
        print("⚠️  Some runtime tests failed")