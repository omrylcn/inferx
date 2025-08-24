# InferX Configuration Guide

✅ **PRODUCTION READY** - Configuration system works with package import and CLI usage

InferX uses a flexible, hierarchical configuration system that works with all usage patterns - whether you're importing InferX as a package or generating projects with templates. This guide explains how to configure InferX for your specific needs.

## 🏗️ Configuration Architecture

### Configuration Priority (Highest to Lowest)

1. **CLI Arguments** - `--device gpu --runtime onnx`
2. **User-specified config** - `--config my_config.yaml`  
3. **Project local config** - `./inferx_config.yaml`
4. **User global config** - `~/.inferx/config.yaml`
5. **Package default config** - Built-in defaults

### Configuration Locations

```
📁 Package default:     inferx/configs/default.yaml
📁 User global:         ~/.inferx/config.yaml  
📁 Project local:       ./inferx_config.yaml
📁 User specified:      --config custom.yaml
```

## 🚀 Quick Start

### 1. Initialize User Configuration

```bash
# Create global user config
inferx init --global

# Or use the config command
inferx config --init
```

This creates `~/.inferx/config.yaml` with a template you can customize.

### 2. Create Project Configuration

```bash
# Create a config template for your project
inferx config --template ./my_project_config.yaml

# Copy and customize
cp ./my_project_config.yaml ./inferx_config.yaml
```

### 3. Validate Configuration

```bash
# Check your configuration for issues
inferx config --validate

# View current active configuration
inferx config --show
```

## ⚙️ Configuration Sections

### Model Detection

Configure how InferX automatically detects model types:

```yaml
model_detection:
  yolo_keywords:
    - "yolo"
    - "yolov8" 
    - "yolov11"
    - "my_custom_yolo"    # Add your model names
  
  classification_keywords:
    - "resnet"
    - "efficientnet"
    - "my_classifier"
```

### Device Mapping

Map friendly device names to runtime-specific names:

```yaml
device_mapping:
  auto: "GPU"              # Prefer GPU when auto selected
  cpu: "CPU"
  gpu: "GPU"
  myriad: "MYRIAD"         # Intel VPU
  hddl: "HDDL"             # Intel HDDL
```

### Model Defaults

Set default parameters for different model types:

```yaml
model_defaults:
  yolo:
    input_size: 640
    confidence_threshold: 0.25
    nms_threshold: 0.45
    class_names:           # Custom class names
      - "person"
      - "vehicle"
      - "my_custom_class"
  
  classification:
    input_size: [224, 224]
    top_k: 5
```

### Performance Presets

Define performance optimization profiles:

```yaml
performance_presets:
  production:
    openvino:
      performance_hint: "THROUGHPUT"
      num_streams: 0       # Auto-optimize
    onnx:
      providers: ["CUDAExecutionProvider"]
      
  development:
    openvino:
      performance_hint: "LATENCY"
      num_streams: 1
```

## 🎯 Common Use Cases

### High-Accuracy YOLO Setup

```yaml
# inferx_config.yaml
model_defaults:
  yolo:
    input_size: 1024       # Higher resolution
    confidence_threshold: 0.1  # Lower threshold
    nms_threshold: 0.3     # Tighter NMS
```

### Production Deployment

```yaml
# production_config.yaml
performance_presets:
  production:
    openvino:
      performance_hint: "THROUGHPUT"
      device: "GPU"
      
logging:
  level: "WARNING"         # Reduce log verbosity
  
output:
  include_metadata:
    timing_info: false     # Reduce output size
```

### Development/Debug Setup

```yaml
# debug_config.yaml
logging:
  level: "DEBUG"
  categories:
    preprocessing: true
    postprocessing: true
    model_loading: true
    
output:
  include_metadata:
    config_used: true      # Show config in output
```

### Custom Model Integration

```yaml
# custom_model_config.yaml
model_detection:
  yolo_keywords:
    - "my_vehicle_detector"
    - "traffic_sign_model"
    
model_defaults:
  yolo:
    class_names:
      - "car"
      - "truck" 
      - "bus"
      - "motorcycle"
      - "stop_sign"
      - "traffic_light"
```

## 🔧 Runtime-Specific Settings

### ONNX Runtime

```yaml
runtime_preferences:
  onnx:
    providers:
      gpu: ["CUDAExecutionProvider", "CPUExecutionProvider"]
      cpu: ["CPUExecutionProvider"]
    
    session_options:
      graph_optimization_level: "ORT_ENABLE_ALL"
      inter_op_num_threads: 4
      intra_op_num_threads: 4
```

### OpenVINO Runtime

```yaml
runtime_preferences:
  openvino:
    performance_hints:
      throughput: "THROUGHPUT"
      latency: "LATENCY"
    
    device_optimizations:
      CPU:
        CPU_BIND_THREAD: "YES"
        CPU_THROUGHPUT_STREAMS: "CPU_THROUGHPUT_AUTO"
      GPU:
        GPU_THROUGHPUT_STREAMS: "GPU_THROUGHPUT_AUTO"
```

## 📝 CLI Usage Examples

### Using Different Configurations

```bash
# Use default configuration
inferx run model.onnx image.jpg

# Use project local config
inferx run model.onnx image.jpg  # Automatically loads ./inferx_config.yaml

# Use specific config file
inferx run model.onnx image.jpg --config production_config.yaml

# Override with CLI arguments
inferx run model.onnx image.jpg --config myconfig.yaml --device gpu --runtime openvino
```

### Configuration Management

```bash
# Initialize user config
inferx config --init

# Create config template
inferx config --template ./my_template.yaml

# Validate current config
inferx config --validate

# Show effective configuration
inferx config --show
```

## 🔍 Configuration Debugging

### Check Active Configuration

```bash
# See what configuration is being used
inferx config --show

# Validate for issues
inferx config --validate
```

### Enable Debug Logging

```yaml
logging:
  level: "DEBUG"
  categories:
    model_loading: true     # See model loading details
    device_info: true       # See device selection
    preprocessing: true     # See preprocessing steps
```

### Include Config in Output

```yaml
output:
  include_metadata:
    config_used: true       # Include config in inference output
```

## 📚 Advanced Configuration

### Model Caching

```yaml
advanced:
  model_cache:
    enabled: true
    cache_dir: "~/.inferx/cache"
    max_cache_size_gb: 5
```

### Memory Management

```yaml
advanced:
  memory:
    enable_memory_pool: true
    max_memory_pool_size_mb: 1024
```

### Experimental Features

```yaml
advanced:
  experimental:
    enable_dynamic_batching: true
    enable_model_optimization: true
    enable_mixed_precision: true
```

## 🐛 Troubleshooting

### Configuration Not Loading

1. Check file exists: `ls -la ./inferx_config.yaml`
2. Validate YAML syntax: `inferx config --validate`  
3. Check permissions: `ls -la ~/.inferx/config.yaml`

### Model Type Not Detected

Add your model name to detection keywords:

```yaml
model_detection:
  yolo_keywords:
    - "your_model_name"
```

### Performance Issues

1. Try different performance presets
2. Adjust device settings
3. Enable model caching
4. Use appropriate runtime (OpenVINO for Intel, ONNX for NVIDIA)

### Device Not Available

Check device mapping and availability:

```yaml
device_mapping:
  your_device: "CORRECT_RUNTIME_NAME"
```

## 📖 Further Reading

- [Default Configuration](inferx/configs/default.yaml) - Complete default settings
- [Example Configuration](examples/example_config.yaml) - Real-world examples
- [API Documentation](docs/api.md) - Programmatic configuration
- [Performance Guide](docs/performance.md) - Optimization tips