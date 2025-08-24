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
- `template` - Generate inference project templates 🆕
- `api` - Add FastAPI server to existing project 🆕
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

## 🎨 Template Generation ✅

**Generate project templates with optional layers:**

### Basic Template Generation
```bash
# Generate YOLO ONNX template
uv run inferx template --model-type yolo --name my_yolo_project

# Generate YOLO OpenVINO template  
uv run inferx template --model-type yolo_openvino --name my_openvino_project
```

### Template with Model Files
```bash
# Generate YOLO ONNX template with model file
uv run inferx template --model-type yolo --name my_yolo_project --model-path /path/to/yolo_model.onnx

# Generate YOLO OpenVINO template with model directory
uv run inferx template --model-type yolo_openvino --name my_openvino_project --model-path /path/to/openvino_model_dir
```

### Template with API Layer
```bash
# Add FastAPI server to template
uv run inferx template --model-type yolo --name my_api_project --with-api

# With model file
uv run inferx template --model-type yolo --name my_api_project --with-api --model-path /path/to/model.onnx
```

### Template with Docker Layer
```bash
# Add Docker container to template
uv run inferx template --model-type yolo --name my_docker_project --with-docker

# With model file
uv run inferx template --model-type yolo --name my_docker_project --with-docker --model-path /path/to/model.onnx
```

### Full-Stack Template
```bash
# Generate complete template with API and Docker
uv run inferx template --model-type yolo --name my_complete_project --with-api --with-docker --model-path /path/to/model.onnx
```

### Template Project Structure
```
my_project/
├── src/
│   ├── __init__.py
│   ├── inferencer.py     # Model-specific inferencer
│   ├── base.py          # Base inferencer class
│   ├── yolo_base.py     # YOLO base functionality
│   ├── utils.py         # Helper functions
│   ├── exceptions.py    # Custom exceptions
│   └── server.py        # FastAPI server (if --with-api)
├── models/
│   ├── yolo_model.onnx  # Model file (if --model-path provided)
│   └── yolo_model.xml   # OpenVINO model (if --model-path provided)
├── data/
│   └── test_image.jpg   # Sample test image
├── config.yaml          # Configuration file
├── pyproject.toml       # Package dependencies
├── Dockerfile           # Docker configuration (if --with-docker)
├── docker-compose.yml   # Docker Compose (if --with-docker)
├── README.md            # Project documentation
└── .gitignore           # Git ignore patterns
```

### Using Generated Templates
```bash
# Navigate to generated project
cd my_project

# Install dependencies
uv pip install -e .

# Test inferencer
uv run python -c "from src.inferencer import Inferencer; print('✅ Inferencer loaded successfully')"

# Run inference on test image
uv run python -c "
from src.inferencer import Inferencer
inferencer = Inferencer('models/yolo_model.onnx')
result = inferencer.predict('data/test_image.jpg')
print(f'Detected {result[\"num_detections\"]} objects')
"

# Run API server (if --with-api)
uv run --extra api python -m src.server

# Build Docker image (if --with-docker)
docker build -t my_project:latest .
```

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
# Generate production-ready template with model
uv run inferx template --model-type yolo_openvino --name production_detector --model-path /path/to/model.xml --with-api --with-docker

# Navigate to project
cd production_detector

# Install dependencies
uv pip install -e .

# Test inferencer
uv run python -c "from src.inferencer import Inferencer; print('✅ Production detector ready')"

# Run API server
uv run --extra api python -m src.server

# Build and run Docker container
docker build -t detector:latest .
docker run -p 8080:8080 detector:latest
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
# Generate YOLO ONNX template with model
uv run inferx template --model-type yolo --name my-yolo-detector --model-path /path/to/model.onnx

# Generate YOLO OpenVINO template with model
uv run inferx template --model-type yolo_openvino --name my-openvino-detector --model-path /path/to/model_dir

# Generate template and add API server
uv run inferx template --model-type yolo --name my-detector --with-api --model-path /path/to/model.onnx

# Generate template and add Docker support
uv run inferx template --model-type yolo --name my-detector --with-docker --model-path /path/to/model.onnx

# Generate complete stack
uv run inferx template --model-type yolo_openvino --name my-complete-detector --with-api --with-docker --model-path /path/to/model_dir
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

**Template Generation Features (Completed):**
- **✅ Project templates**: `inferx template --model-type yolo --name my-detector`
- **✅ OpenVINO templates**: `inferx template --model-type yolo_openvino --name my-detector`
- **✅ FastAPI server**: `inferx template --model-type yolo --name my-detector --with-api`
- **✅ Docker generation**: `inferx template --model-type yolo --name my-detector --with-docker`
- **✅ Full stack templates**: `inferx template --model-type yolo --name my-detector --with-api --with-docker`
- **✅ Model file copying**: `inferx template --model-type yolo --name my-detector --model-path /path/to/model.onnx`
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

*InferX v1.0 - Production-ready dual-runtime ML inference package with full OpenVINO support and template generation! 🚀*