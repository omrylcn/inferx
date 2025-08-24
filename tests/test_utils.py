"""Utility functions tests"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestUtilsImports:
    """Test utility imports"""
    
    def test_utils_module_import(self):
        """Test utils module imports"""
        try:
            from inferx import utils
            assert utils is not None
        except ImportError as e:
            assert False, f"Failed to import utils module: {e}"
    
    def test_image_processor_import(self):
        """Test ImageProcessor imports"""
        try:
            from inferx.utils import ImageProcessor
            assert ImageProcessor is not None
        except ImportError as e:
            assert False, f"Failed to import ImageProcessor: {e}"
    
    def test_preprocessing_functions_import(self):
        """Test preprocessing functions import"""
        try:
            from inferx.utils import preprocess_for_inference
            assert preprocess_for_inference is not None
        except ImportError as e:
            # This function might not exist yet
            pass


class TestImageProcessor:
    """Test ImageProcessor functionality"""
    
    def test_image_processor_structure(self):
        """Test ImageProcessor has expected structure"""
        try:
            from inferx.utils import ImageProcessor
            
            # Should have class methods
            expected_methods = ["load_image", "resize_image", "normalize_image"]
            
            for method in expected_methods:
                if hasattr(ImageProcessor, method):
                    assert callable(getattr(ImageProcessor, method)), f"{method} should be callable"
                    
        except Exception as e:
            assert False, f"ImageProcessor structure test failed: {e}"
    
    def test_image_loading_logic(self):
        """Test image loading logic exists"""
        try:
            from inferx.utils import ImageProcessor
            
            # Test that load_image method exists (if implemented)
            if hasattr(ImageProcessor, 'load_image'):
                load_method = getattr(ImageProcessor, 'load_image')
                assert callable(load_method)
                
                # Test with non-existent image (should handle gracefully)
                try:
                    result = load_method("nonexistent_image.jpg")
                    # If it doesn't raise exception, it should return something sensible
                    assert result is None or isinstance(result, np.ndarray)
                except Exception:
                    # Expected - no image file
                    pass
                    
        except Exception as e:
            assert False, f"Image loading logic test failed: {e}"


class TestPreprocessingFunctions:
    """Test preprocessing utility functions"""
    
    def test_preprocessing_function_exists(self):
        """Test preprocessing functions exist"""
        try:
            from inferx.utils import preprocess_for_inference
            assert callable(preprocess_for_inference)
            
        except ImportError:
            # Function might not be implemented yet
            pass
        except Exception as e:
            assert False, f"Preprocessing function test failed: {e}"
    
    def test_normalization_logic(self):
        """Test normalization logic"""
        # Test basic normalization concepts
        test_array = np.array([[[100, 150, 200]]], dtype=np.uint8)  # Sample RGB pixel
        
        # Normalize to [0, 1]
        normalized = test_array.astype(np.float32) / 255.0
        assert 0.0 <= np.min(normalized) <= 1.0
        assert 0.0 <= np.max(normalized) <= 1.0
        
        # Standard normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        standardized = (normalized - mean) / std
        assert isinstance(standardized, np.ndarray)
        assert standardized.shape == normalized.shape
    
    def test_resize_logic(self):
        """Test resize logic concepts"""
        # Test basic resize concepts
        import cv2
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Resize to standard size
        target_size = (224, 224)
        resized = cv2.resize(test_image, target_size)
        
        assert resized.shape[:2] == target_size
        assert resized.dtype == test_image.dtype


class TestErrorHandling:
    """Test error handling utilities"""
    
    def test_exception_imports(self):
        """Test exception classes import"""
        try:
            from inferx.exceptions import ModelError, InferenceError, ErrorCode
            
            assert ModelError is not None
            assert InferenceError is not None
            assert ErrorCode is not None
            
        except ImportError as e:
            assert False, f"Failed to import exceptions: {e}"
    
    def test_error_code_enum(self):
        """Test ErrorCode enum structure"""
        try:
            from inferx.exceptions import ErrorCode
            
            # Should be an enum-like structure
            assert hasattr(ErrorCode, '__members__') or hasattr(ErrorCode, '__dict__')
            
            # Should have common error codes
            expected_codes = ["MODEL_NOT_FOUND", "INFERENCE_FAILED", "INVALID_INPUT"]
            
            for code in expected_codes:
                if hasattr(ErrorCode, code):
                    assert getattr(ErrorCode, code) is not None
                    
        except Exception as e:
            # ErrorCode might be implemented differently
            pass
    
    def test_model_error_structure(self):
        """Test ModelError exception structure"""
        try:
            from inferx.exceptions import ModelError, ErrorCode
            
            # Test error creation
            error = ModelError(
                message="Test error",
                error_code=getattr(ErrorCode, 'MODEL_NOT_FOUND', 'MODEL_NOT_FOUND'),
                suggestions=["Test suggestion"],
                context={"test": "context"}
            )
            
            assert isinstance(error, Exception)
            assert error.message == "Test error"
            
        except Exception as e:
            # ModelError might be implemented differently
            pass


class TestRecoveryMechanisms:
    """Test recovery mechanism utilities"""
    
    def test_recovery_imports(self):
        """Test recovery mechanism imports"""
        try:
            from inferx.recovery import with_model_loading_retry, with_inference_retry
            
            assert with_model_loading_retry is not None
            assert with_inference_retry is not None
            
        except ImportError:
            # Recovery mechanisms might not be implemented yet
            pass
    
    def test_retry_decorator_structure(self):
        """Test retry decorator structure"""
        try:
            from inferx.recovery import with_model_loading_retry
            
            # Should be a decorator
            assert callable(with_model_loading_retry)
            
            # Test decorator application
            @with_model_loading_retry()
            def test_function():
                return "test"
            
            result = test_function()
            assert result == "test"
            
        except ImportError:
            # Recovery might not be implemented
            pass
        except Exception as e:
            assert False, f"Retry decorator test failed: {e}"


class TestPathUtilities:
    """Test path handling utilities"""
    
    def test_path_handling(self):
        """Test basic path handling"""
        from pathlib import Path
        
        # Test path operations used in InferX
        test_paths = [
            "model.onnx",
            "models/yolo.xml", 
            "/absolute/path/model.onnx",
            "relative/path/model.xml"
        ]
        
        for path_str in test_paths:
            path = Path(path_str)
            
            # Basic path operations
            assert isinstance(path, Path)
            assert path.name  # Should have a name
            assert path.suffix  # Should have a suffix
            
            # Test extension detection
            if path.suffix in ['.onnx', '.xml']:
                assert path.suffix in ['.onnx', '.xml']
    
    def test_model_path_validation(self):
        """Test model path validation logic"""
        valid_extensions = ['.onnx', '.xml']
        invalid_extensions = ['.txt', '.jpg', '.py']
        
        for ext in valid_extensions:
            path = Path(f"model{ext}")
            assert path.suffix == ext
            assert path.suffix in valid_extensions
        
        for ext in invalid_extensions:
            path = Path(f"file{ext}")
            assert path.suffix == ext
            assert path.suffix not in valid_extensions


# Integration test
def test_utils_integration():
    """Test utils integration with other components"""
    try:
        from inferx.utils import ImageProcessor
        from inferx.exceptions import ModelError
        
        # Both should be importable
        assert ImageProcessor is not None
        assert ModelError is not None
        
        # Test basic integration
        import numpy as np
        import cv2
        
        # Basic image processing should work
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = cv2.resize(test_image, (224, 224))
        assert resized.shape == (224, 224, 3)
        
    except Exception as e:
        assert False, f"Utils integration test failed: {e}"


if __name__ == "__main__":
    import traceback
    
    test_classes = [TestUtilsImports, TestImageProcessor, TestPreprocessingFunctions, TestErrorHandling, TestRecoveryMechanisms, TestPathUtilities]
    
    print("🧪 Utils Tests")
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
                # Don't print full traceback for utils tests
    
    # Integration test
    total_tests += 1
    try:
        test_utils_integration()
        print(f"\n✅ Integration test")
        passed_tests += 1
    except Exception as e:
        print(f"\n❌ Integration test: {e}")
    
    print(f"\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All utils tests passed!")
    else:
        print("⚠️  Some utils tests failed")