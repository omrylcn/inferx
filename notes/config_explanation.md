# InferX Config System Explained

## 🎯 Config Dosyalarının Rolleri

### 1. Default Config (`inferx/configs/default.yaml`)
**Purpose**: System-wide defaults ve global settings
**Scope**: Tüm InferX için temel ayarlar
**Contents**: 
- Model detection keywords
- Device mappings
- Runtime preferences
- Performance presets
- Default thresholds

```yaml
# default.yaml - Sistem geneli varsayılanlar
model_detection:
  yolo_keywords: ["yolo", "yolov8", "ultralytics"]
  classification_keywords: ["resnet", "efficientnet"]

device_mapping:
  auto: "AUTO"
  cpu: "CPU"
  gpu: "GPU"

model_defaults:
  yolo:
    confidence_threshold: 0.25
    input_size: 640
```

### 2. Template Config (`templates/yolo/config.yaml`)
**Purpose**: Generated project için başlangıç ayarları
**Scope**: Specific template (YOLO, classification, etc.)
**Contents**:
- Model-specific settings
- Project structure references
- Template defaults

```yaml
# templates/yolo/config.yaml - YOLO template için
model:
  path: "models/yolo_model.onnx"  # Template'deki model path
  type: "yolo"                    # Template type

inference:
  device: "auto"                  # Override edilebilir
  runtime: "auto"
  confidence_threshold: 0.25      # YOLO specific
  input_size: 640

preprocessing:
  target_size: [640, 640]         # YOLO preprocessing
  maintain_aspect_ratio: true
```

### 3. User Global Config (`~/.inferx/config.yaml`)
**Purpose**: Kullanıcının genel tercihleri
**Scope**: Tüm projects için user preferences
**Created by**: `inferx config --init`

```yaml
# ~/.inferx/config.yaml - User preferences
device_mapping:
  auto: "GPU"  # User prefers GPU by default

model_defaults:
  yolo:
    confidence_threshold: 0.3  # User prefers lower threshold

logging:
  level: "DEBUG"  # User wants verbose logging
```

### 4. Project Local Config (`./inferx_config.yaml`)
**Purpose**: Project-specific ayarlar
**Scope**: Tek project için custom settings
**Version Control**: Git'e commit edilir

```yaml
# ./inferx_config.yaml - Project specific
model_defaults:
  yolo:
    input_size: 1024  # This project needs high resolution
    class_names:      # Custom classes for this project
      - "car"
      - "truck"
      - "motorcycle"

advanced:
  model_cache:
    cache_dir: "./cache"  # Project local cache
```

### 5. Runtime Config (`--config custom.yaml`)
**Purpose**: Specific run için ayarlar
**Scope**: Tek çalıştırma için temporary settings
**Use Case**: Testing, experiments, production runs

```yaml
# custom.yaml - Specific run config
inference:
  device: "cpu"           # Force CPU for this run
  batch_size: 1           # Single image processing
  confidence_threshold: 0.5  # Higher threshold for this test
```

## 🔄 Config Merge Process

### Nasıl Çalışır:
```python
# config.py:24-55
def _load_config_hierarchy(self):
    # 1. Start with default config
    self._config = self._load_default_config()
    
    # 2. Merge user global config
    global_config = ~/.inferx/config.yaml
    if exists: merge(self._config, global_config)
    
    # 3. Merge project local config  
    project_config = ./inferx_config.yaml
    if exists: merge(self._config, project_config)
    
    # 4. Merge user-specified config
    if --config provided: merge(self._config, user_config)
    
    # 5. CLI args override everything
```

### Merge Örneği:
```yaml
# default.yaml
model_defaults:
  yolo:
    confidence_threshold: 0.25
    input_size: 640
    nms_threshold: 0.45

# ~/.inferx/config.yaml (user global)
model_defaults:
  yolo:
    confidence_threshold: 0.3  # Override default

# ./inferx_config.yaml (project local)
model_defaults:
  yolo:
    input_size: 1024  # Override for this project

# Final merged result:
model_defaults:
  yolo:
    confidence_threshold: 0.3   # From user global
    input_size: 1024            # From project local
    nms_threshold: 0.45         # From default
```

## 🎭 Template Config'in Özel Rolü

### Template Generation Sırasında:
```python
# generators/template.py:110-155
def _update_config_for_model_type(self, model_type, project_path):
    config_file = project_path / "config.yaml"
    
    if model_type == "yolo_openvino":
        # Template'deki config.yaml'ı güncelle
        config_content = config_content.replace(
            'path: "models/yolo_model.onnx"', 
            'path: "models/yolo_model.xml"'
        )
        # Model type'ı güncelle
        config_content = config_content.replace(
            'type: "yolo"', 
            'type: "yolo_openvino"'
        )
```

### Template Config'in Faydaları:
1. **Ready-to-use**: Generated project hemen çalışır
2. **Model-specific**: Her template kendi optimized settings
3. **Customizable**: User istediği gibi değiştirebilir
4. **Documentation**: Settings'lerin ne anlama geldiğini gösterir

## 📊 Config Kullanım Senaryoları

### Senaryo 1: Development
```bash
# Developer kendi global settings yapar
inferx config --init  # Creates ~/.inferx/config.yaml

# Project'e specific settings ekler
echo "model_defaults:\n  yolo:\n    input_size: 1024" > inferx_config.yaml

# Run with project settings
inferx run model.onnx image.jpg  # Uses merged config
```

### Senaryo 2: Production
```bash
# Production config hazırlar
cat > production.yaml << EOF
inference:
  device: "gpu"
  batch_size: 8
logging:
  level: "WARNING"
EOF

# Production run
inferx run model.xml images/ --config production.yaml
```

### Senaryo 3: Template Usage
```bash
# YOLO template generate eder
inferx template yolo --name my-detector

# Generated config.yaml:
cd my-detector
cat config.yaml
# model:
#   path: "models/yolo_model.onnx"
#   type: "yolo"
# inference:
#   confidence_threshold: 0.25

# User customize eder
sed -i 's/0.25/0.3/' config.yaml
```

## 🔍 Config Debugging

### Config Hierarchy Debug:
```bash
# Show current effective config
inferx config --show

# Validate config
inferx config --validate

# See which configs are loaded
inferx run model.onnx image.jpg --verbose
# Output:
# Loading default config: /path/to/default.yaml
# Merging user config: ~/.inferx/config.yaml  
# Merging project config: ./inferx_config.yaml
# Final device: GPU (from user config)
```

### Config Values Trace:
```python
# config.py debugging
logger.debug(f"Loading default config: {default_config_path}")
logger.debug(f"Merging user config: {user_global_config}")
logger.debug(f"Final device: {config.get('device_mapping.auto')}")
```

## 🎯 Best Practices

### 1. Default Config (System)
- **Never modify directly** - Git managed
- Contains sensible defaults
- Comprehensive documentation
- Backward compatibility preserved

### 2. User Global Config
- **Personal preferences** only
- Device preferences
- Logging levels  
- Performance presets

### 3. Project Local Config
- **Project-specific** settings
- Custom class names
- Model paths
- **Commit to git**

### 4. Runtime Config
- **Temporary** settings
- Testing configurations
- Environment-specific (dev/prod)
- **Don't commit**

## 💡 Key Benefits

1. **Flexibility**: 5-level hierarchy için maximum customization
2. **Inheritance**: Lower levels inherit from higher levels
3. **Override**: Her level bir üstünü override edebilir
4. **Portability**: Templates ready-to-use configs sağlar
5. **Maintainability**: Default config merkezi olarak yönetilir
6. **User Experience**: Sane defaults, easy customization

Bu sistem sayesinde InferX hem novice hem expert users için esnek ve güçlü config management sağlar.