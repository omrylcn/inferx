# InferX Test Suite

Comprehensive but simple tests for InferX functionality.

## Test Files:

- `test_settings.py` - Settings system and configuration tests
- `test_runtime.py` - Runtime engine and model loading tests  
- `test_yolo.py` - YOLO-specific functionality tests
- `test_cli.py` - CLI command tests
- `test_template.py` - Template generation tests
- `test_utils.py` - Utility function tests
- `run_tests.py` - Simple standalone test runner

## Usage:

```bash
# Simple runner (no external dependencies)
python tests/run_tests.py

# Pytest (if available)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_settings.py -v
```

## Test Categories:

### Core Functionality
- ✅ Settings loading and validation
- ✅ Model type detection
- ✅ Runtime engine creation
- ✅ Device mapping

### YOLO Features  
- ✅ YOLO preprocessing
- ✅ YOLO postprocessing logic
- ✅ YOLO configuration defaults

### CLI Features
- ✅ Basic CLI commands
- ✅ Help text generation
- ✅ Argument parsing

### Template System
- ✅ Template generation logic
- ✅ Project creation
- ✅ Configuration handling

### Utilities
- ✅ Image processing
- ✅ Error handling
- ✅ Path handling

## Philosophy:

- ✅ **Simple**: Easy to understand and maintain
- ✅ **Fast**: Quick execution without heavy mocking
- ✅ **Practical**: Tests real functionality
- ✅ **Comprehensive**: Covers all major features
- ✅ **Reliable**: Consistent and stable results