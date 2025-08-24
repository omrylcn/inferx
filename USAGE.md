# InferX Usage Guide

## 🎯 Current State: Production-Ready Package with CLI Interface

InferX is now **production-ready** as both a minimal dependency ML inference package and CLI tool! You can either import InferX directly in your Python code or use it from the command line. This guide covers all available features including the new OpenVINO integration and advanced configuration system.

**Latest Update**: ✅ **Full OpenVINO Support** - YOLO models now work with both ONNX Runtime and OpenVINO Runtime, with automatic model type detection and optimized hardware acceleration for Intel devices.

## 📦 Installation

### Using UV (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install InferX
git clone <repository-url>
cd inferx

# Install in development mode with all dependencies
uv sync --all-extras

# Verify installation
uv run inferx --version
```

### Using pip (Alternative)

```bash
# From project directory
pip install -e .

# With optional dependencies
pip install -e .[gpu,serve,dev]

# Verify installation
inferx --version
```

## 🎯 Two Primary Usage Patterns (Both Production Ready)

### 1. **📦 Package Usage** - Import directly in Python code
```python
from inferx import InferenceEngine

# Use directly in your Python applications
engine = InferenceEngine("model.onnx", device="gpu")
result = engine.predict("image.jpg")
print(result)

# Batch processing
results = engine.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### 2. **⚡ CLI Usage** - Run from command line
```bash
# Run inference directly from command line
uv run inferx run model.onnx image.jpg --device gpu

# Batch processing
uv run inferx run model.xml images/ --output results.json --runtime openvino
```

### Quick Start with UV

```bash
# All commands use 'uv run' prefix
uv run inferx run model.onnx image.jpg
```

## 🎯 Core Features Available

### ✅ **Dual Runtime Support:**
- **ONNX Runtime inference** - Load and run any ONNX model (.onnx files)
- **OpenVINO Runtime inference** - Load and run OpenVINO models (.xml/.bin files) with full YOLO support ✅
- **Automatic runtime selection** - Chooses optimal runtime based on model format and hardware ✅

### ✅ **Model Support:**
- **YOLO object detection** - Both ONNX and OpenVINO versions with shared optimizations ✅
- **Generic model inference** - Support for any ONNX or OpenVINO model
- **Smart model detection** - Automatically detects model type from filename and extension ✅

### ✅ **Processing Capabilities:**
- **Single image processing** - Process individual images
- **Batch processing** - Process entire folders of images with progress tracking
- **Advanced preprocessing** - Letterboxing, normalization, color format conversion
- **Multiple output formats** - JSON/YAML results export

### ✅ **Configuration & Performance:**
- **Hierarchical configuration** - Global, project, and user-specified configs ✅
- **Performance presets** - Throughput, latency, and balanced optimization modes ✅
- **Device flexibility** - CPU, GPU, MYRIAD, HDDL, NPU support with Intel optimizations ✅
- **Memory management** - Model caching and memory pooling ✅

### ✅ **Developer Tools:**
- **Configuration management** - Init, validate, and show config commands ✅
- **Performance tracking** - Detailed timing information
- **Verbose logging** - Debug and troubleshooting support
- **Config validation** - Automatic validation with helpful warnings ✅

## 🛠️ CLI Commands

### Basic Command Structure
```bash
inferx [OPTIONS] COMMAND [ARGS]...
```

### Global Options
- `--verbose, -v`: Enable detailed logging
- `--version`: Show version and exit
- `--help`: Show help message

### Available Commands
- `run` - Run inference on models
- `config` - Configuration management 🆕
- `serve` - Start API server (template generation feature)
- `docker` - Generate Docker containers (template generation feature)
- `init` - Initialize projects or configs 🆕

## 📊 Running Inference

### 1. Single Image Inference

**Basic ONNX usage (UV):**
```bash
uv run inferx run model.onnx image.jpg
```

**Basic OpenVINO usage (UV):** ✅
```bash
uv run inferx run model.xml image.jpg
```

**YOLO object detection:**
```bash
# ONNX YOLO (auto-detected by filename)
uv run inferx run yolov8n.onnx image.jpg

# OpenVINO YOLO (auto-detected by filename and extension) ✅
uv run inferx run yolov8n.xml image.jpg

# Force specific runtime
uv run inferx run yolov8.onnx image.jpg --runtime openvino
uv run inferx run yolov8.xml image.jpg --runtime onnx
```

**Device selection:** ✅
```bash
# Auto-select best device
uv run inferx run model.xml image.jpg --device auto

# Intel CPU optimization
uv run inferx run model.xml image.jpg --device cpu --runtime openvino

# Intel GPU (iGPU)
uv run inferx run model.xml image.jpg --device gpu --runtime openvino

# Intel VPU (Myriad)
uv run inferx run model.xml image.jpg --device myriad --runtime openvino
```

**Save results:**
```bash
uv run inferx run model.xml image.jpg --output results.json
```

**Alternative (pip install):**
```bash
inferx run model.xml image.jpg
```

**Example output (Generic ONNX):**
```
🚀 Starting inference...
   Model: model.onnx
   Input: image.jpg
   Device: auto, Runtime: auto
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running single image inference...
✅ Inference completed in 0.032s

📊 Inference Summary:
   Model type: onnx_generic
   Outputs: 1
   Inference time: 0.032s
```

**Example output (YOLO ONNX):**
```
🚀 Starting inference...
   Model: yolov8n.onnx
   Input: image.jpg
   Device: auto, Runtime: onnx
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running single image inference...
✅ Inference completed in 0.032s

📊 Inference Summary:
   Model type: yolo_onnx
   Detections: 3
   Inference time: 0.032s
```

**Example output (YOLO OpenVINO):** ✅
```
🚀 Starting inference...
   Model: yolov8n.xml
   Input: image.jpg
   Device: CPU, Runtime: openvino
⏳ Loading model...
✅ Model loaded in 0.180s
🔍 Running single image inference...
✅ Inference completed in 0.025s

📊 Inference Summary:
   Model type: yolo_openvino
   Detections: 4
   Inference time: 0.025s
```

### 2. Batch Processing

**Process entire folder (UV):**
```bash
uv run inferx run model.onnx photos/
```

**With progress tracking (UV):**
```bash
uv run inferx run model.onnx photos/ --output batch_results.json --verbose
```

**Alternative (pip install):**
```bash
inferx run model.onnx photos/
```

**Example output:**
```
🚀 Starting inference...
   Model: model.onnx
   Input: photos/
   Device: auto, Runtime: auto
⏳ Loading model...
✅ Model loaded in 0.245s
🔍 Running batch inference on 25 images...
Processing images  [####################################]  100%
✅ Batch processing completed!
   Processed: 25/25 images
   Total time: 0.850s
   Average: 0.034s per image
```

### 3. Configuration Management ✅

**Initialize user configuration:**
```bash
# Create global user config at ~/.inferx/config.yaml
uv run inferx config --init

# Or use init command
uv run inferx init --global
```

**Create project configuration:**
```bash
# Create config template
uv run inferx config --template my_config.yaml

# Copy to project directory
cp my_config.yaml inferx_config.yaml
```

**Configuration validation and inspection:**
```bash
# Validate current configuration
uv run inferx config --validate

# Show active configuration
uv run inferx config --show
```

**Example config.yaml:** ✅
```yaml
# Model detection (add your custom patterns)
model_detection:
  yolo_keywords:
    - "yolo"
    - "yolov8" 
    - "my_custom_yolo"

# Device preferences
device_mapping:
  auto: "GPU"  # Prefer GPU when auto is selected

# Performance optimization
performance_presets:
  production:
    openvino:
      performance_hint: "THROUGHPUT"
      num_streams: 0  # Auto-optimize
    onnx:
      providers: ["CUDAExecutionProvider"]

# Model-specific settings
model_defaults:
  yolo:
    confidence_threshold: 0.3    # Lower threshold
    input_size: 1024            # Higher resolution
    class_names:                # Custom classes
      - "person"
      - "vehicle"
      - "my_custom_class"

# Preprocessing pipeline
preprocessing_defaults:
  openvino:
    target_size: [640, 640]
    normalize: true
    color_format: "RGB"
```

**Use configuration:**
```bash
# Use project local config (auto-loaded from ./inferx_config.yaml)
uv run inferx run model.xml image.jpg

# Use specific config file
uv run inferx run model.xml image.jpg --config production.yaml

# Override with CLI arguments
uv run inferx run model.xml image.jpg --config myconfig.yaml --device gpu
```

## 🎛️ Available Options

### Device Selection ✅
```bash
# Automatic device selection (default)
uv run inferx run model.xml image.jpg --device auto

# CPU inference (Intel optimization)
uv run inferx run model.xml image.jpg --device cpu

# GPU inference (Intel iGPU or NVIDIA)
uv run inferx run model.xml image.jpg --device gpu

# Intel VPU (Myriad stick)
uv run inferx run model.xml image.jpg --device myriad

# Intel HDDL (High Density Deep Learning)
uv run inferx run model.xml image.jpg --device hddl

# Neural Processing Unit
uv run inferx run model.xml image.jpg --device npu
```

### Runtime Selection ✅
```bash
# Automatic runtime selection (default)
uv run inferx run model.xml image.jpg --runtime auto

# Force ONNX Runtime
uv run inferx run model.xml image.jpg --runtime onnx

# Force OpenVINO Runtime
uv run inferx run model.onnx image.jpg --runtime openvino

# Cross-format inference (ONNX model on OpenVINO)
uv run inferx run yolov8.onnx image.jpg --runtime openvino
```

### Output Formats (UV)
```bash
# JSON output (default)
uv run inferx run model.onnx image.jpg --output results.json --format json

# YAML output
uv run inferx run model.onnx image.jpg --output results.yaml --format yaml
```

### Verbose Mode (UV)
```bash
# Enable detailed logging
uv run inferx run model.onnx image.jpg --verbose

# See full results in terminal
uv run inferx run model.onnx image.jpg -v
```

## 📄 Output Format

### Single Image Results (Generic ONNX)
```json
{
  "raw_outputs": [...],
  "output_shapes": [[1, 1000]],
  "num_outputs": 1,
  "model_type": "onnx_generic",
  "output_0_shape": [1, 1000],
  "output_0_dtype": "float32",
  "output_0_min": -4.2,
  "output_0_max": 8.1,
  "output_0_mean": 0.05,
  "timing": {
    "model_load_time": 0.245,
    "inference_time": 0.032,
    "total_time": 0.277
  }
}
```

### Single Image Results (YOLO)
```json
{
  "detections": [
    {
      "bbox": [100.5, 150.2, 200.0, 300.0],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "person"
    },
    {
      "bbox": [300.1, 200.5, 150.0, 250.0],
      "confidence": 0.87,
      "class_id": 1,
      "class_name": "bicycle"
    }
  ],
  "num_detections": 2,
  "model_type": "yolo",
  "timing": {
    "model_load_time": 0.245,
    "inference_time": 0.032,
    "total_time": 0.277
  }
}
```

### Batch Processing Results
```json
{
  "batch_summary": {
    "total_images": 25,
    "successful": 25,
    "failed": 0,
    "total_inference_time": 0.850,
    "average_inference_time": 0.034,
    "model_load_time": 0.245
  },
  "results": [
    {
      "file_path": "/path/to/image1.jpg",
      "inference_time": 0.032,
      "raw_outputs": [...],
      "model_type": "onnx_generic"
    },
    ...
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

**1. Model loading fails:**
```bash
❌ Failed to load ONNX model: No such file or directory
```
- Check if model file exists and path is correct
- Ensure model is in ONNX format (.onnx extension)

**2. Image loading fails:**
```bash
❌ Could not load image: /path/to/image.jpg
```
- Verify image file exists and is readable
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp

**3. GPU not available:**
```bash
GPU requested but no GPU providers available, falling back to CPU
```
- Install ONNX Runtime GPU: `pip install onnxruntime-gpu`
- Check CUDA installation and compatibility

### Debug Mode

**Enable verbose logging (UV):**
```bash
uv run inferx run model.onnx image.jpg --verbose
```

**Alternative (pip install):**
```bash
inferx run model.onnx image.jpg --verbose
```

This will show:
- Detailed model loading information
- Provider selection process
- Input/output shapes and types
- Full error stack traces
- Complete results JSON

## 📊 Performance Tips

### 1. Device Optimization (UV)
```bash
# Use GPU for better performance (if available)
uv run inferx run model.onnx image.jpg --device gpu

# Use CPU with specific configuration
uv run inferx run model.onnx image.jpg --device cpu --config cpu_optimized.yaml
```

### 2. Batch Processing (UV)
```bash
# Process multiple images efficiently
uv run inferx run model.onnx photos/ --batch-size 8
```

### 3. Configuration Tuning
```yaml
# config.yaml - Performance optimized
runtime:
  device: "gpu"
  
session_options:
  graph_optimization_level: "ORT_ENABLE_ALL"
  inter_op_num_threads: 4
  intra_op_num_threads: 4
```

## 🔄 Model Requirements

### Supported Model Formats
- **ONNX models** (.onnx files)
- Any model exported from PyTorch, TensorFlow, etc.

### Input Requirements  
- **Image inputs**: Any standard image format
- **Preprocessing**: Automatic resizing, normalization, color format conversion
- **Batch dimension**: Automatically added if needed

### Output Processing
- **Raw outputs**: Numerical arrays from model
- **Basic statistics**: Min, max, mean values
- **Shape information**: Input/output tensor shapes
- **Timing data**: Performance metrics

## 🚀 Example Workflows

### 1. Quick Model Testing
```bash
# Test ONNX model quickly
uv run inferx run my_model.onnx test_image.jpg --verbose

# Test OpenVINO model quickly ✅
uv run inferx run my_model.xml test_image.jpg --verbose
```

### 2. Performance Optimization ✅
```bash
# Compare ONNX vs OpenVINO performance
uv run inferx run yolov8.onnx test_image.jpg --device cpu --verbose
uv run inferx run yolov8.xml test_image.jpg --device cpu --verbose

# Test different devices with OpenVINO
uv run inferx run model.xml test_image.jpg --device cpu --verbose
uv run inferx run model.xml test_image.jpg --device gpu --verbose
uv run inferx run model.xml test_image.jpg --device myriad --verbose
```

### 3. Production Deployment Setup ✅
```bash
# Create production configuration
uv run inferx config --template production_config.yaml

# Edit config for your use case, then test
uv run inferx run model.xml images/ --config production_config.yaml

# Validate configuration
uv run inferx config --validate
```

### 4. Batch Evaluation
```bash
# Process validation dataset with ONNX
uv run inferx run model.onnx validation_images/ --output validation_results.json

# Process with OpenVINO for better performance ✅
uv run inferx run model.xml validation_images/ --device gpu --output validation_results.json
```

### 5. Custom Model Integration ✅
```bash
# Add your model detection pattern
echo "model_detection:" > custom_config.yaml
echo "  yolo_keywords:" >> custom_config.yaml
echo "    - 'my_vehicle_detector'" >> custom_config.yaml

# Test with custom configuration
uv run inferx run my_vehicle_detector.xml image.jpg --config custom_config.yaml
```

### 6. Template Generation Workflows ✅
```bash
# Generate YOLO ONNX template
uv run inferx template yolo --name my-yolo-detector

# Generate YOLO OpenVINO template with model copy
uv run inferx template yolo_openvino --name my-openvino-detector --model-path /path/to/model.xml

# Generate template and add API server
uv run inferx template yolo --name my-detector --with-api

# Generate template and add Docker support
uv run inferx template yolo --name my-detector --with-docker
```

### 7. Development and Debugging ✅
```bash
# Enable debug logging
uv run inferx run model.xml image.jpg --verbose

# Show what configuration is being used
uv run inferx config --show

# Validate your configuration
uv run inferx config --validate
```

---

## 🎯 What's Coming Next

**Template Generation Features (Partially Completed):**
- **✅ Project templates**: `inferx template yolo --name my-detector`
- **✅ OpenVINO templates**: `inferx template yolo_openvino --name my-detector`
- **✅ FastAPI server**: `inferx api` (adds server.py to existing project)
- **✅ Docker generation**: `inferx docker model.xml --tag mymodel:v1 --runtime openvino`
- **Performance benchmarking**: Built-in benchmarking tools for optimization
- **Advanced testing**: Comprehensive unit and integration test suite

**Phase 3 - Advanced Model Support:**
- **🚧 Anomalib integration**: Full support for anomaly detection models (ONNX + OpenVINO)
- **🚧 Classification models**: ResNet, EfficientNet, MobileNet support with auto-detection
- **Segmentation models**: U-Net, DeepLab, SegFormer support

**Phase 4 - Ecosystem & Deployment:**
- **Model zoo integration**: Pre-trained model downloads and management
- **Cloud deployment**: AWS, Azure, GCP deployment guides
- **Edge optimization**: Raspberry Pi, Jetson, Intel NUC optimization guides
- **WebUI**: Browser-based model testing and configuration interface

---

## 🌟 **Current Achievement Summary**

✅ **Dual Runtime Support** - ONNX Runtime + OpenVINO Runtime with full YOLO support  
✅ **Smart Model Detection** - Automatic model type detection from filenames and extensions  
✅ **Multi-Device Support** - CPU, GPU, MYRIAD, HDDL, NPU compatibility with Intel optimizations  
✅ **Production Configuration** - Hierarchical config system with validation  
✅ **Performance Optimization** - Hardware-specific optimizations and presets  
✅ **Developer Tools** - Configuration management and debugging utilities  
✅ **Package Usage** - Import and use directly in Python code  
✅ **CLI Usage** - Run models directly from command line  
✅ **Template Generation** - Generate standalone projects with YOLO template  
✅ **API Generation** - Add FastAPI server to existing projects  
✅ **Docker Generation** - Generate optimized Docker deployment  

*InferX v1.0 - Production-ready dual-runtime ML inference package with full OpenVINO support! 🚀*