#!/usr/bin/env python3
"""
Basic InferX Usage Examples

This script demonstrates the fundamental ways to use InferX for inference
with both ONNX and OpenVINO models.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inferx import InferenceEngine


def get_test_image(image_name="street_scene.jpg"):
    """Get path to test image from data directory"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    test_image_path = project_root / "data" / "test_images" / image_name
    
    if test_image_path.exists():
        print(f"✅ Using real test image: {test_image_path}")
        return test_image_path
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
        print("   Run: mkdir -p data/test_images && curl -o data/test_images/street_scene.jpg 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop'")
        
        # Fallback: create dummy image
        return create_dummy_image(Path("dummy_image.jpg"))

def create_dummy_image(output_path: Path, size=(640, 480)):
    """Create a dummy test image as fallback"""
    # Create random test image
    image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    # Add some shapes to make it more interesting
    cv2.rectangle(image, (100, 100), (300, 200), (0, 255, 0), 2)
    cv2.circle(image, (400, 300), 50, (0, 0, 255), -1)
    
    cv2.imwrite(str(output_path), image)
    print(f"✅ Created fallback test image: {output_path}")
    return output_path


def example_basic_onnx_inference():
    """Example 1: Basic ONNX model inference"""
    print("\n🔥 Example 1: Basic ONNX Inference")
    print("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model file (for demonstration)
            model_path = temp_path / "test_model.onnx"
            model_path.touch()
            
            # Use real test image
            image_path = get_test_image("street_scene.jpg")
            
            print(f"📄 Model: {model_path}")
            print(f"🖼️  Image: {image_path}")
            
            # Note: This would fail with a real inference engine since we don't have a real model
            # But shows the basic usage pattern
            print("""
💡 Usage Pattern:
   from inferx import InferenceEngine
   
   engine = InferenceEngine("model.onnx")
   result = engine.predict("image.jpg")
   
   print(f"Results: {result}")
""")
            
            print("✅ Basic ONNX inference pattern demonstrated")
            
    except Exception as e:
        print(f"ℹ️  Note: {e}")
        print("   This is expected since we're using a dummy model file")


def example_openvino_inference():
    """Example 2: OpenVINO model inference"""
    print("\n🔥 Example 2: OpenVINO Inference")
    print("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy OpenVINO model files
            xml_path = temp_path / "model.xml"
            bin_path = temp_path / "model.bin"
            xml_path.touch()
            bin_path.touch()
            
            # Use real test image
            image_path = get_test_image("street_scene.jpg")
            
            print(f"📄 Model: {xml_path}")
            print(f"🖼️  Image: {image_path}")
            
            print("""
💡 Usage Pattern:
   from inferx import InferenceEngine
   
   # OpenVINO model with device selection
   engine = InferenceEngine(
       model_path="model.xml",
       device="cpu",        # cpu, gpu, myriad, auto
       runtime="openvino"   # force OpenVINO runtime
   )
   
   result = engine.predict("image.jpg")
   print(f"OpenVINO Results: {result}")
""")
            
            print("✅ OpenVINO inference pattern demonstrated")
            
    except Exception as e:
        print(f"ℹ️  Note: {e}")


def example_yolo_detection():
    """Example 3: YOLO Object Detection"""
    print("\n🔥 Example 3: YOLO Object Detection")
    print("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy YOLO model (auto-detected by filename)
            yolo_onnx = temp_path / "yolov8n.onnx"
            yolo_openvino = temp_path / "yolov8n.xml"
            
            yolo_onnx.touch()
            yolo_openvino.touch()
            (temp_path / "yolov8n.bin").touch()  # OpenVINO needs .bin file
            
            # Use real test image with objects
            image_path = get_test_image("cars_traffic.jpg")
            
            print(f"📄 YOLO ONNX: {yolo_onnx}")
            print(f"📄 YOLO OpenVINO: {yolo_openvino}")
            print(f"🖼️  Image: {image_path}")
            
            print("""
💡 YOLO Usage Pattern:
   # ONNX YOLO (automatically detected)
   engine = InferenceEngine("yolov8n.onnx")
   result = engine.predict("street_scene.jpg")
   
   # Expected result format:
   {
     "detections": [
       {
         "bbox": [100, 150, 200, 300],    # [x, y, width, height]
         "confidence": 0.95,
         "class_id": 0,
         "class_name": "person"
       }
     ],
     "num_detections": 1,
     "model_type": "yolo_onnx"
   }
   
   # OpenVINO YOLO (better performance on Intel hardware)
   engine = InferenceEngine("yolov8n.xml", device="cpu")
   result = engine.predict("street_scene.jpg")
""")
            
            print("✅ YOLO detection patterns demonstrated")
            
    except Exception as e:
        print(f"ℹ️  Note: {e}")


def example_batch_processing():
    """Example 4: Batch Processing"""
    print("\n🔥 Example 4: Batch Processing")
    print("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model
            model_path = temp_path / "batch_model.onnx"
            model_path.touch()
            
            # Create multiple test images
            image_dir = temp_path / "images"
            image_dir.mkdir()
            
            # Use real test images for batch processing
            available_images = ["street_scene.jpg", "person_bicycle.jpg", "cars_traffic.jpg"]
            image_paths = []
            for img_name in available_images:
                img_path = get_test_image(img_name)
                image_paths.append(img_path)
            
            print(f"📄 Model: {model_path}")
            print(f"📁 Images: {len(image_paths)} images in {image_dir}")
            
            print("""
💡 Batch Processing Pattern:
   from inferx import InferenceEngine
   
   engine = InferenceEngine("model.onnx")
   
   # Method 1: Process list of image paths
   image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
   results = engine.predict_batch(image_paths)
   
   # Method 2: Use CLI for batch processing
   # inferx run model.onnx images/ --output batch_results.json
   
   print(f"Processed {len(results)} images")
   for i, result in enumerate(results):
       print(f"Image {i}: {result['model_type']}")
""")
            
            print("✅ Batch processing patterns demonstrated")
            
    except Exception as e:
        print(f"ℹ️  Note: {e}")


def example_configuration_usage():
    """Example 5: Configuration Usage"""
    print("\n🔥 Example 5: Configuration Usage")
    print("=" * 50)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config file
            config_path = temp_path / "custom_config.yaml"
            config_content = """
# Custom InferX Configuration with Pydantic Settings
model_detection:
  yolo_keywords:
    - "yolo"
    - "my_detector"

device_mapping:
  auto: "GPU"  # Prefer GPU

model_defaults:
  yolo:
    confidence_threshold: 0.3    # Lower threshold for more detections
    input_size: 1024            # Higher resolution (must be divisible by 32)
    class_names:
      - "person"
      - "vehicle" 
      - "custom_object"

logging:
  level: "INFO"
  model_loading: true
  inference_timing: true
"""
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            print(f"📄 Config: {config_path}")
            
            print("""
💡 Configuration Usage Pattern:
   # Method 1: Use config file with InferenceEngine
   engine = InferenceEngine(
       "my_detector.xml", 
       config_path="custom_config.yaml"
   )
   
   # Method 2: CLI with config
   # inferx run my_detector.xml image.jpg --config custom_config.yaml
   
   # Method 3: Project-level config (automatic)
   # Save as: ./inferx_config.yaml
   # Will be automatically loaded
   
   # Method 4: User global config
   # inferx config --init  # Creates ~/.inferx/config.yaml
   
   # Method 5: Config management with Pydantic validation
   # inferx config --show      # View current config with validation
   # inferx config --validate  # Pydantic type checking
""")
            
            print("✅ Configuration usage patterns demonstrated")
            
    except Exception as e:
        print(f"ℹ️  Note: {e}")


def example_device_optimization():
    """Example 6: Device-Specific Optimization"""
    print("\n🔥 Example 6: Device-Specific Optimization")
    print("=" * 50)
    
    print("""
💡 Device Optimization Patterns:

# Intel CPU Optimization
engine = InferenceEngine(
    "model.xml", 
    device="cpu",
    runtime="openvino"
)

# Intel GPU (iGPU) Optimization  
engine = InferenceEngine(
    "model.xml",
    device="gpu", 
    runtime="openvino"
)

# Intel VPU (Myriad USB Stick)
engine = InferenceEngine(
    "model.xml",
    device="myriad",
    runtime="openvino" 
)

# NVIDIA GPU with ONNX Runtime
engine = InferenceEngine(
    "model.onnx",
    device="gpu",
    runtime="onnx"
)

# CLI Examples:
# inferx run model.xml image.jpg --device cpu --runtime openvino
# inferx run model.xml image.jpg --device gpu 
# inferx run model.xml image.jpg --device myriad
# inferx run model.onnx image.jpg --device gpu --runtime onnx
""")
    
    print("✅ Device optimization patterns demonstrated")


def main():
    """Run all usage examples"""
    print("🚀 InferX Usage Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use InferX")
    print("Note: Examples use dummy model files for demonstration")
    
    examples = [
        example_basic_onnx_inference,
        example_openvino_inference, 
        example_yolo_detection,
        example_batch_processing,
        example_configuration_usage,
        example_device_optimization
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
            print("   This is expected when using dummy model files")
    
    print("\n🎯 Summary")
    print("=" * 50)
    print("✅ InferX supports:")
    print("   - ONNX Runtime & OpenVINO Runtime")
    print("   - YOLO object detection (auto-detected)")
    print("   - Single & batch processing")
    print("   - Hierarchical configuration system")
    print("   - Multi-device support (CPU, GPU, VPU)")
    print("   - Cross-runtime inference")
    
    print("\n📚 Next Steps:")
    print("   1. Install InferX: pip install -e .")
    print("   2. Get real models (ONNX or OpenVINO format)")
    print("   3. Initialize config: inferx config --init")
    print("   4. Run inference: inferx run model.onnx image.jpg")
    print("   5. Check documentation: see CONFIG_GUIDE.md")


if __name__ == "__main__":
    main()