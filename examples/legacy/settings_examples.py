#!/usr/bin/env python3
"""
Settings Examples - Pydantic Configuration System

This script demonstrates the new Pydantic-based configuration system,
showing type-safe configuration with validation.
"""

import sys
import tempfile
import json
import yaml
from pathlib import Path

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inferx.settings import get_inferx_settings, validate_yolo_template_config, YoloTemplateConfigSettings


def example_basic_settings():
    """Example 1: Basic Settings Usage"""
    print("\n🔥 Example 1: Basic Settings with Pydantic")
    print("-" * 50)
    
    try:
        # Load settings with automatic validation
        settings = get_inferx_settings()
        
        print("✅ Settings loaded with Pydantic validation")
        print(f"   YOLO input size: {settings.yolo_input_size}")
        print(f"   YOLO confidence: {settings.yolo_confidence_threshold}")
        print(f"   Device mapping: {dict(list(settings.device_mapping.items())[:3])}...")
        print(f"   Log level: {settings.log_level}")
        
        # Show model type detection
        test_files = [
            Path("yolov8n.onnx"),
            Path("yolov8n.xml"), 
            Path("resnet50.onnx"),
        ]
        
        print("\n🔍 Model Type Detection:")
        for file_path in test_files:
            model_type = settings.detect_model_type(file_path)
            print(f"   {file_path.name} → {model_type}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_environment_variables():
    """Example 2: Environment Variable Configuration"""
    print("\n🔥 Example 2: Environment Variables")
    print("-" * 50)
    
    print("💡 Environment Variable Examples:")
    print("   # Set YOLO input size")
    print("   export INFERX_YOLO_INPUT_SIZE=1024")
    print("   ")
    print("   # Set confidence threshold")  
    print("   export INFERX_YOLO_CONFIDENCE_THRESHOLD=0.3")
    print("   ")
    print("   # Set log level")
    print("   export INFERX_LOG_LEVEL=DEBUG")
    print("   ")
    print("   # Set device preference")
    print("   export INFERX_DEVICE_AUTO=GPU")
    
    print("\n✅ Environment variables automatically loaded with INFERX_ prefix")


def example_yolo_template_validation():
    """Example 3: YOLO Template Validation"""
    print("\n🔥 Example 3: YOLO Template Config Validation")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid YOLO template config
            valid_config_path = temp_path / "valid_yolo.yaml"
            valid_config = {
                "model": {
                    "path": "models/yolo_model.onnx",
                    "type": "yolo"
                },
                "inference": {
                    "device": "auto",
                    "runtime": "auto",
                    "confidence_threshold": 0.25,
                    "nms_threshold": 0.45,
                    "input_size": 640,
                    "batch_size": 1
                },
                "preprocessing": {
                    "target_size": [640, 640],
                    "normalize": True,
                    "color_format": "RGB",
                    "maintain_aspect_ratio": True
                }
            }
            
            with open(valid_config_path, 'w') as f:
                yaml.dump(valid_config, f, default_flow_style=False)
            
            print(f"✅ Created valid YOLO config: {valid_config_path}")
            
            try:
                # Validate with Pydantic
                validated = validate_yolo_template_config(valid_config_path)
                print("✅ YOLO config validation successful!")
                print(f"   Model path: {validated.model_path}")
                print(f"   Input size: {validated.input_size}")
                print(f"   Confidence: {validated.confidence_threshold}")
                print(f"   Device: {validated.device}")
                
            except ImportError:
                print("⚠️  pydantic-settings not available, skipping validation")
            
            # Create invalid config to show validation
            invalid_config_path = temp_path / "invalid_yolo.yaml"
            invalid_config = {
                "model": {
                    "path": "",  # Invalid: empty path
                    "type": "yolo"
                },
                "inference": {
                    "confidence_threshold": 1.5,  # Invalid: > 1.0
                    "input_size": 100  # Invalid: not divisible by 32
                }
            }
            
            with open(invalid_config_path, 'w') as f:
                yaml.dump(invalid_config, f, default_flow_style=False)
            
            print(f"\n❌ Testing invalid config: {invalid_config_path}")
            try:
                validate_yolo_template_config(invalid_config_path)
                print("   Unexpected: validation should have failed")
            except ValueError as e:
                print(f"   ✅ Validation correctly caught errors: {str(e)[:100]}...")
            except ImportError:
                print("   ⚠️  pydantic-settings not available for validation")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_config_hierarchy():
    """Example 4: Configuration Hierarchy"""
    print("\n🔥 Example 4: Configuration Hierarchy")
    print("-" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample config files
            default_config = temp_path / "default.yaml"
            default_config.write_text("""
# Default Configuration
model_defaults:
  yolo:
    input_size: 640
    confidence_threshold: 0.25

device_mapping:
  auto: "AUTO"
  cpu: "CPU"
  gpu: "GPU"
""")
            
            user_config = temp_path / "user_config.yaml"
            user_config.write_text("""
# User Override Configuration
model_defaults:
  yolo:
    confidence_threshold: 0.3  # User prefers higher confidence
    
device_mapping:
  auto: "GPU"  # User prefers GPU
""")
            
            print(f"✅ Configuration Hierarchy:")
            print(f"   1. Default: {default_config}")
            print(f"   2. User:    {user_config}")
            
            print(f"\n📊 Settings loads configs in priority order:")
            print(f"   Priority: CLI args > User config > Project config > Global config > Defaults")
            
            # Show current settings
            settings = get_inferx_settings()
            print(f"\n🎯 Current Effective Settings:")
            print(f"   YOLO confidence: {settings.yolo_confidence_threshold}")
            print(f"   YOLO input size: {settings.yolo_input_size}")
            print(f"   Auto device: {settings.device_mapping.get('auto')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def example_type_safety():
    """Example 5: Type Safety Benefits"""
    print("\n🔥 Example 5: Type Safety with Pydantic")
    print("-" * 50)
    
    print("💡 Type Safety Benefits:")
    print("   ✅ IDE auto-completion")
    print("   ✅ Runtime type checking")
    print("   ✅ Automatic validation")
    print("   ✅ Clear error messages")
    
    try:
        settings = get_inferx_settings()
        
        print(f"\n🎯 Type-Safe Access:")
        print(f"   settings.yolo_input_size        → {settings.yolo_input_size} (int)")
        print(f"   settings.yolo_confidence_threshold → {settings.yolo_confidence_threshold} (float)")
        print(f"   settings.device_mapping         → dict with {len(settings.device_mapping)} keys")
        print(f"   settings.log_level             → {settings.log_level} (enum)")
        
        print(f"\n🔧 Model-Specific Defaults:")
        yolo_defaults = settings.get_model_defaults("yolo")
        print(f"   YOLO defaults: {len(yolo_defaults)} settings")
        print(f"   Input size: {yolo_defaults.get('input_size')}")
        print(f"   Confidence: {yolo_defaults.get('confidence_threshold')}")
        
        print(f"\n🖥️  Device Mapping:")
        for device, mapping in settings.device_mapping.items():
            print(f"   {device} → {mapping}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_practical_usage():
    """Example 6: Practical Usage Patterns"""
    print("\n🔥 Example 6: Practical Usage Patterns")
    print("-" * 50)
    
    print("💡 Common Usage Patterns:")
    
    print(f"\n1. 📦 Direct Python Usage:")
    print(f"   from inferx.settings import get_inferx_settings")
    print(f"   settings = get_inferx_settings()")
    print(f"   print(f'YOLO size: {{settings.yolo_input_size}}')")
    
    print(f"\n2. 🛠️ Template Generation:")
    print(f"   inferx template yolo --name my-detector")
    print(f"   # Automatically validates generated config.yaml")
    
    print(f"\n3. ⚙️ CLI Configuration:")
    print(f"   inferx config --show      # Show current settings")
    print(f"   inferx config --validate  # Validate with Pydantic")
    
    print(f"\n4. 🌍 Environment Variables:")
    print(f"   export INFERX_YOLO_INPUT_SIZE=1024")
    print(f"   export INFERX_LOG_LEVEL=DEBUG")
    print(f"   # Automatically loaded by settings.py")
    
    print(f"\n5. 📁 Project Configuration:")
    print(f"   # Create inferx_config.yaml in project root")
    print(f"   # Settings automatically loaded from hierarchy")


def main():
    """Run all settings examples"""
    print("⚙️  InferX Settings Examples - Pydantic Configuration")
    print("=" * 70)
    print("This script demonstrates the new type-safe configuration system")
    
    examples = [
        example_basic_settings,
        example_environment_variables,
        example_yolo_template_validation,
        example_config_hierarchy,
        example_type_safety,
        example_practical_usage
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
    
    print(f"\n🎯 Settings System Summary")
    print("=" * 50)
    print("✅ InferX Settings Features:")
    print("   - Type-safe configuration with Pydantic")
    print("   - Environment variable support (INFERX_* prefix)")
    print("   - YAML hierarchy loading (default → user → project)")
    print("   - Template validation for generated projects")
    print("   - Automatic validation with helpful error messages")
    print("   - IDE auto-completion and type checking")
    print("   - Backward compatibility with existing API")
    
    print(f"\n📚 Key Benefits:")
    print("   🔒 Type Safety - No more config typos or wrong types")
    print("   🎯 Validation - Catch errors early with clear messages")
    print("   🌍 Environment - Easy deployment configuration")
    print("   📝 Documentation - Self-documenting with type hints")
    print("   🚀 Performance - Cached loading and validation")
    
    print(f"\n🚀 Migration from old config.py:")
    print("   ✅ COMPLETED - config.py replaced with settings.py")
    print("   ✅ COMPATIBLE - Existing API works without changes")
    print("   ✅ ENHANCED - Added validation and type safety")


if __name__ == "__main__":
    main()