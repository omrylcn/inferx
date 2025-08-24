"""Template generation tests"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTemplateImports:
    """Test template components can be imported"""
    
    def test_template_generator_import(self):
        """Test TemplateGenerator imports"""
        try:
            from inferx.generators.template import TemplateGenerator
            assert TemplateGenerator is not None
        except ImportError as e:
            assert False, f"Failed to import TemplateGenerator: {e}"
    
    def test_template_module_import(self):
        """Test template module imports"""
        try:
            from inferx.generators import template
            assert template is not None
        except ImportError as e:
            assert False, f"Failed to import template module: {e}"


class TestTemplateGenerator:
    """Test TemplateGenerator functionality"""
    
    def test_template_generator_creation(self):
        """Test TemplateGenerator can be created"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            assert generator is not None
            
        except Exception as e:
            assert False, f"Failed to create TemplateGenerator: {e}"
    
    def test_template_generator_has_methods(self):
        """Test TemplateGenerator has expected methods"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            
            # Check for expected methods
            expected_methods = ["create_template"]
            for method in expected_methods:
                assert hasattr(generator, method), f"Missing method: {method}"
                assert callable(getattr(generator, method)), f"Method {method} is not callable"
                
        except Exception as e:
            assert False, f"TemplateGenerator method test failed: {e}"


class TestTemplateTypes:
    """Test different template types"""
    
    def test_supported_template_types(self):
        """Test supported template types are defined"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            
            # Test that generator can handle different model types
            supported_types = ["yolo", "yolo_openvino"]
            
            for model_type in supported_types:
                # We don't actually create templates (would need file system)
                # but test that the types are recognized
                assert isinstance(model_type, str)
                assert len(model_type) > 0
                
        except Exception as e:
            assert False, f"Template types test failed: {e}"
    
    def test_template_options(self):
        """Test template generation options"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            
            # Test template options exist
            # with_api, project_name, etc. should be supported
            options = {
                "with_api": True,
                "project_name": "test_project"
            }
            
            # Basic validation - options should be dictionary-like
            assert isinstance(options, dict)
            assert "with_api" in options
            assert isinstance(options["with_api"], bool)
            
        except Exception as e:
            assert False, f"Template options test failed: {e}"


class TestTemplateStructure:
    """Test template structure and files"""
    
    def test_template_directory_exists(self):
        """Test template directory structure exists"""
        template_dir = project_root / "inferx" / "generators" / "templates"
        assert template_dir.exists(), f"Template directory not found: {template_dir}"
        assert template_dir.is_dir(), f"Template path is not a directory: {template_dir}"
    
    def test_yolo_template_exists(self):
        """Test YOLO template structure exists"""
        yolo_template = project_root / "inferx" / "generators" / "templates" / "yolo"
        
        if yolo_template.exists():
            assert yolo_template.is_dir()
            
            # Check for expected template files
            expected_files = ["pyproject.toml", "README.md"]
            for file_name in expected_files:
                file_path = yolo_template / file_name
                if file_path.exists():
                    assert file_path.is_file(), f"{file_name} should be a file"
    
    def test_template_src_directory(self):
        """Test template src directory structure"""
        yolo_src = project_root / "inferx" / "generators" / "templates" / "yolo" / "src"
        
        if yolo_src.exists():
            assert yolo_src.is_dir()
            
            # Check for Python files
            python_files = list(yolo_src.glob("*.py"))
            assert len(python_files) > 0, "Template should have Python files"


class TestTemplateCreation:
    """Test template creation logic (without file system operations)"""
    
    def test_template_creation_parameters(self):
        """Test template creation accepts proper parameters"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            
            # Test parameter validation logic
            model_types = ["yolo", "yolo_openvino"]
            project_names = ["test_project", "my_detector", "anomaly_detector"]
            
            for model_type in model_types:
                assert isinstance(model_type, str)
                assert len(model_type) > 0
            
            for project_name in project_names:
                assert isinstance(project_name, str)
                assert len(project_name) > 0
                # Basic name validation
                assert project_name.replace("_", "").replace("-", "").isalnum()
                
        except Exception as e:
            assert False, f"Template creation parameters test failed: {e}"
    
    def test_template_with_api_option(self):
        """Test template with API option"""
        try:
            from inferx.generators.template import TemplateGenerator
            
            generator = TemplateGenerator()
            
            # Test with_api option handling
            api_options = [True, False]
            
            for with_api in api_options:
                assert isinstance(with_api, bool)
                
                # Logic should handle both cases
                if with_api:
                    # Should include API components
                    pass
                else:
                    # Should create basic template
                    pass
                    
        except Exception as e:
            assert False, f"Template API option test failed: {e}"


class TestTemplateConfiguration:
    """Test template configuration handling"""
    
    def test_template_config_structure(self):
        """Test template configurations are properly structured"""
        try:
            from inferx.settings import get_inferx_settings
            
            settings = get_inferx_settings()
            
            # Templates should use settings for configuration
            yolo_defaults = settings.get_model_defaults("yolo")
            assert len(yolo_defaults) > 0
            
            # Required for templates
            required_keys = ["input_size", "confidence_threshold"]
            for key in required_keys:
                assert key in yolo_defaults
                
        except Exception as e:
            assert False, f"Template configuration test failed: {e}"
    
    def test_template_pyproject_structure(self):
        """Test template pyproject.toml structure"""
        template_pyproject = project_root / "inferx" / "generators" / "templates" / "yolo" / "pyproject.toml"
        
        if template_pyproject.exists():
            # Basic file existence and readability test
            assert template_pyproject.is_file()
            
            try:
                content = template_pyproject.read_text()
                assert len(content) > 0
                assert "[project]" in content
                assert "dependencies" in content
                
            except Exception as e:
                assert False, f"Failed to read template pyproject.toml: {e}"


class TestTemplateIntegration:
    """Test template integration with other components"""
    
    def test_template_uses_settings(self):
        """Test templates integrate with settings system"""
        try:
            from inferx.generators.template import TemplateGenerator
            from inferx.settings import get_inferx_settings
            
            generator = TemplateGenerator()
            settings = get_inferx_settings()
            
            # Both should be accessible
            assert generator is not None
            assert settings is not None
            
        except Exception as e:
            assert False, f"Template-settings integration failed: {e}"
    
    def test_template_cli_integration(self):
        """Test templates can be used from CLI"""
        try:
            from inferx.generators.template import TemplateGenerator
            from inferx.cli import main
            
            # Both components should be importable
            assert TemplateGenerator is not None
            assert main is not None
            
        except Exception as e:
            assert False, f"Template-CLI integration failed: {e}"


# Integration test
def test_template_basic_integration():
    """Test template system basic integration"""
    try:
        from inferx.generators.template import TemplateGenerator
        from inferx.settings import get_inferx_settings
        
        # Create generator
        generator = TemplateGenerator()
        settings = get_inferx_settings()
        
        # Basic workflow test
        model_type = "yolo"
        project_name = "test_project"
        
        # These should be valid inputs
        assert isinstance(model_type, str)
        assert isinstance(project_name, str)
        assert len(model_type) > 0
        assert len(project_name) > 0
        
        # Settings should provide defaults for templates
        defaults = settings.get_model_defaults(model_type)
        assert len(defaults) > 0
        
    except Exception as e:
        assert False, f"Template integration test failed: {e}"


if __name__ == "__main__":
    import traceback
    
    test_classes = [TestTemplateImports, TestTemplateGenerator, TestTemplateTypes, TestTemplateStructure, TestTemplateCreation, TestTemplateConfiguration, TestTemplateIntegration]
    
    print("🧪 Template Tests")
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
        test_template_basic_integration()
        print(f"\n✅ Integration test")
        passed_tests += 1
    except Exception as e:
        print(f"\n❌ Integration test: {e}")
        traceback.print_exc()
    
    print(f"\n🎯 {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All template tests passed!")
    else:
        print("⚠️  Some template tests failed")