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
- `serve` - Start standalone API server 🆕
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

## 🎨 Template Generation ✅ **FULLY WORKING**

**Generate project templates - 4 combinations available:**

### 4 Template Combinations
```bash
# 1. YOLO ONNX (Basic)
uv run inferx template --model-type yolo --name my_yolo_basic

# 2. YOLO ONNX (with API)
uv run inferx template --model-type yolo --name my_yolo_api --with-api

# 3. YOLO OpenVINO (Basic)  
uv run inferx template --model-type yolo_openvino --name my_openvino_basic

# 4. YOLO OpenVINO (with API)
uv run inferx template --model-type yolo_openvino --name my_openvino_api --with-api
```

### Template with Model Files
```bash
# Copy your model during generation
uv run inferx template --model-type yolo --name my_detector --model-path /path/to/model.onnx

# OpenVINO with model files (.xml + .bin)
uv run inferx template --model-type yolo_openvino --name my_openvino --model-path /path/to/model.xml
```

### Template Setup & Usage
```bash
# After generating template:
cd my_yolo_api

# Install dependencies
uv sync --extra api          # For API templates
uv sync --extra openvino     # For OpenVINO templates  
uv sync                      # For basic templates

# Test inferencer
uv run python -c "from src.inferencer import Inferencer; print('✅ Ready!')"

# Start API server (API templates only)
uv run --extra api python -m src.server
```

### Template Project Structure ✅
```
my_yolo_api/                    # Generated project
├── src/
│   ├── __init__.py
│   ├── inferencer.py          # YOLO inference implementation
│   ├── base.py               # Base inferencer class
│   ├── yolo_base.py          # YOLO base functionality
│   ├── utils.py              # Image processing utilities
│   ├── exceptions.py         # Custom exceptions
│   └── server.py             # FastAPI server (if --with-api)
├── models/
│   └── yolo_model.onnx       # Place your model here (.xml for OpenVINO)
├── config.yaml               # Model configuration
├── pyproject.toml            # UV-compatible dependencies
├── README.md                 # Usage instructions
└── .gitignore                # Standard Python gitignore
```

**Key Features:**
- ✅ **UV-compatible** - `uv sync` installs dependencies
- ✅ **No relative imports** - Works with `uv run`
- ✅ **Dynamic project names** - Uses your `--name`
- ✅ **Optional dependencies** - `--extra api` for FastAPI
- ✅ **OpenVINO support** - Auto-includes openvino dependency

### Using Generated Templates ✅
```bash
# Navigate to generated project
cd my_yolo_api

# Install dependencies (choose appropriate command)
uv sync                      # Basic templates
uv sync --extra api          # API templates  
uv sync --extra openvino     # OpenVINO templates

# Test inferencer import (should work immediately)
uv run python -c "from src.inferencer import Inferencer; print('✅ Inferencer imported!')"

# Test with your model (place model in models/ first)
uv run python -c "
from src.inferencer import Inferencer
import os
if os.path.exists('models/yolo_model.onnx'):
    inferencer = Inferencer('models/yolo_model.onnx')
    print('✅ Model loaded successfully')
else:
    print('📁 Place your model in models/ directory')
"

# Start API server (API templates only)
uv run --extra api python -m src.server
# Server available at: http://0.0.0.0:8080

# Test API endpoints
curl -X GET "http://localhost:8080/"
curl -X POST "http://localhost:8080/predict" -F "file=@your_image.jpg"
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
# Compare ONNX vs OpenVINO performance with YOLOv11n
uv run inferx -v run models/yolo11n_onnx/yolo11n.onnx data/person.jpeg
uv run inferx -v run models/yolo11n_openvino/yolo11n.xml data/person.jpeg

# Test different devices with OpenVINO
uv run inferx run models/yolo11n_openvino/yolo11n.xml data/person.jpeg --device cpu
uv run inferx run models/yolo11n_openvino/yolo11n.xml data/person.jpeg --device gpu
uv run inferx run models/yolo11n_openvino/yolo11n.xml data/person.jpeg --device auto
```

**Performance Comparison Results (YOLOv11n on person.jpeg):**
| Runtime | Model Load | Inference | Total | Detection Accuracy |
|---------|------------|-----------|-------|-------------------|
| ONNX | 0.048s | 0.128s | 0.177s | 96.4% confidence |
| OpenVINO | 0.352s | 0.090s | 0.442s | 96.4% confidence |

**Key Insights:**
- **ONNX**: Faster model loading, good for development
- **OpenVINO**: Faster inference, better for production on Intel hardware
- **Both**: Identical detection accuracy and results

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

### 6. Template Generation Workflows ✅ **UPDATED**
```bash
# Quick YOLO project setup
uv run inferx template --model-type yolo --name quick_detector
cd quick_detector && uv sync
uv run python -c "from src.inferencer import Inferencer; print('✅ Ready!')"

# Production API setup
uv run inferx template --model-type yolo_openvino --name production_api --with-api
cd production_api && uv sync --extra api
uv run --extra api python -m src.server &
curl -X GET "http://localhost:8080/"

# Complete workflow with model
uv run inferx template --model-type yolo --name my_detector --with-api --model-path /path/to/model.onnx
cd my_detector
uv sync --extra api
# Your model is already copied to models/
uv run --extra api python -m src.server

# All 4 template combinations
uv run inferx template --model-type yolo --name basic_yolo                    # Basic ONNX
uv run inferx template --model-type yolo --name api_yolo --with-api           # ONNX + API
uv run inferx template --model-type yolo_openvino --name basic_openvino       # Basic OpenVINO  
uv run inferx template --model-type yolo_openvino --name api_openvino --with-api  # OpenVINO + API
```

### 7. API Server Workflows ✅
```bash
# Add API server to existing project
uv run inferx api

# Start standalone API server (template-based projects)
cd my_project
uv run --extra api python -m src.server

# Start standalone API server with configuration
cd my_project  
uv run --extra api python -m src.server --host 0.0.0.0 --port 8080
```

### 7. API Server Workflows ✅
```bash
# Start standalone API server with ONNX YOLO model
uv run inferx serve --model-path models/yolo11n_onnx/yolo11n.onnx --model-type yolo --host 127.0.0.1 --port 8080

# Start standalone API server with OpenVINO YOLO model  
uv run inferx serve --model-path models/yolo11n_openvino/yolo11n.xml --model-type yolo --host 127.0.0.1 --port 8080

# Start standalone API server with configuration
uv run inferx serve --config production.yaml

# Start standalone API server with custom host/port
uv run inferx serve --model-path /path/to/model.onnx --model-type yolo --host 0.0.0.0 --port 8080

# Template API server (after template generation)
cd my_project
uv run --extra api python -m src.server
```

**API Endpoints:**
```bash
# Health check
curl -X GET "http://127.0.0.1:8080/"

# Model information
curl -X GET "http://127.0.0.1:8080/info"

# Single image prediction
curl -X POST "http://127.0.0.1:8080/predict" -F "file=@data/person.jpeg"

# Batch prediction
curl -X POST "http://127.0.0.1:8080/predict-batch" -F "files=@image1.jpg" -F "files=@image2.jpg"
```

**Example API Response:**
```json
{
  "detections": [
    {
      "bbox": [74.0, 408.0, 754.0, 743.0],
      "confidence": 0.9636614322662354,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "num_detections": 1,
  "num_outputs": 1,
  "model_type": "yolo"
}
```

### 8. Development and Debugging ✅
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
- **✅ Standalone API server**: `uv run inferx serve --model-path /path/to/model.onnx --model-type yolo`
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
✅ **Standalone API Server** - FastAPI server with proper model type detection **[FIXED]**  
✅ **Docker Generation** - Generate optimized Docker deployment  
✅ **API Endpoints** - `/predict`, `/predict-batch`, `/info`, health checks  
✅ **Consistent Runtime** - Both CLI and API use same `runtime.py` inference engine  

**Recent Updates:**
- ✅ **Template System Complete** - 4 working combinations (YOLO, YOLO+API, OpenVINO, OpenVINO+API)
- ✅ **UV Integration Perfect** - `uv sync` with proper dependency management
- ✅ **Import System Fixed** - No more relative import issues in templates
- ✅ **API Templates Working** - FastAPI servers generate and run successfully
- ✅ **Dynamic Project Names** - Templates use your specified `--name`
- ✅ **OpenVINO Auto-Dependencies** - Automatically includes openvino for OpenVINO templates

*InferX v1.1 - Production-ready ML inference package with fully working template generation system! 🚀*