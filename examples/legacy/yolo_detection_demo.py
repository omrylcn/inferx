#!/usr/bin/env python3
"""
YOLO Detection Demo

This script demonstrates YOLO object detection with InferX,
showing both ONNX and OpenVINO runtime usage.
"""

import sys
import argparse
from pathlib import Path
import tempfile
import numpy as np
import cv2
import json

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inferx import InferenceEngine
from inferx.settings import get_inferx_settings


def create_test_image_with_objects(output_path: Path, size=(1280, 720)):
    """Create a test image with various objects for YOLO detection"""
    print(f"🎨 Creating test image with objects...")
    
    # Create base image (street scene simulation)
    image = np.random.randint(50, 100, (size[1], size[0], 3), dtype=np.uint8)
    
    # Add sky gradient
    for y in range(size[1] // 3):
        color_intensity = int(100 + (y / (size[1] // 3)) * 100)
        image[y, :] = [135, 206, color_intensity]  # Sky blue gradient
    
    # Add ground
    image[size[1] * 2 // 3:, :] = [34, 139, 34]  # Forest green ground
    
    # Simulate objects that YOLO might detect
    
    # Person (rectangle representing a person)
    cv2.rectangle(image, (200, 300), (280, 600), (255, 200, 100), -1)  # Body
    cv2.circle(image, (240, 250), 30, (255, 220, 177), -1)  # Head
    cv2.putText(image, "PERSON", (180, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Car (rectangle representing a car)
    cv2.rectangle(image, (400, 450), (650, 580), (0, 0, 255), -1)  # Car body
    cv2.rectangle(image, (430, 420), (480, 450), (100, 100, 100), -1)  # Window
    cv2.rectangle(image, (570, 420), (620, 450), (100, 100, 100), -1)  # Window
    cv2.circle(image, (430, 580), 25, (64, 64, 64), -1)  # Wheel
    cv2.circle(image, (620, 580), 25, (64, 64, 64), -1)  # Wheel
    cv2.putText(image, "CAR", (480, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Bicycle (simple bicycle shape)
    cv2.circle(image, (800, 520), 40, (0, 255, 255), 3)  # Front wheel
    cv2.circle(image, (900, 520), 40, (0, 255, 255), 3)  # Back wheel
    cv2.line(image, (800, 520), (850, 480), (0, 255, 255), 3)  # Frame
    cv2.line(image, (850, 480), (900, 520), (0, 255, 255), 3)  # Frame
    cv2.line(image, (850, 480), (850, 450), (0, 255, 255), 3)  # Handlebar post
    cv2.putText(image, "BIKE", (820, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Traffic light (simple representation)
    cv2.rectangle(image, (1100, 200), (1150, 350), (64, 64, 64), -1)  # Pole
    cv2.rectangle(image, (1080, 100), (1170, 200), (128, 128, 128), -1)  # Light box
    cv2.circle(image, (1125, 130), 15, (0, 255, 0), -1)  # Green light
    cv2.putText(image, "LIGHT", (1060, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add some texture to make it more realistic
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(output_path), image)
    print(f"✅ Created test image: {output_path}")
    print(f"   Size: {size[0]}x{size[1]}")
    print(f"   Objects: Person, Car, Bicycle, Traffic Light")
    
    return output_path


def simulate_yolo_inference(image_path: Path, model_type: str = "yolo_onnx"):
    """Simulate YOLO inference results (since we don't have real models)"""
    print(f"🔍 Simulating {model_type} inference on {image_path.name}")
    
    # Simulate realistic YOLO detection results
    detections = [
        {
            "bbox": [200, 250, 80, 350],  # [x, y, width, height] - Person
            "confidence": 0.92,
            "class_id": 0,
            "class_name": "person"
        },
        {
            "bbox": [400, 420, 250, 160],  # Car
            "confidence": 0.87,
            "class_id": 2, 
            "class_name": "car"
        },
        {
            "bbox": [760, 480, 140, 80],   # Bicycle
            "confidence": 0.78,
            "class_id": 1,
            "class_name": "bicycle"
        },
        {
            "bbox": [1080, 100, 90, 250],  # Traffic light
            "confidence": 0.65,
            "class_id": 9,
            "class_name": "traffic light"
        }
    ]
    
    result = {
        "detections": detections,
        "num_detections": len(detections),
        "model_type": model_type,
        "timing": {
            "preprocessing_time": 0.005,
            "inference_time": 0.032 if "onnx" in model_type else 0.028,  # OpenVINO slightly faster
            "postprocessing_time": 0.008,
            "total_time": 0.045 if "onnx" in model_type else 0.041
        }
    }
    
    return result


def draw_detections_on_image(image_path: Path, detections: list, output_path: Path):
    """Draw detection results on image"""
    print(f"🎨 Drawing detections on image...")
    
    image = cv2.imread(str(image_path))
    
    # Define colors for different classes
    colors = [
        (0, 255, 0),    # Green for person
        (255, 0, 0),    # Blue for bicycle  
        (0, 0, 255),    # Red for car
        (255, 255, 0),  # Cyan for traffic light
    ]
    
    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["class_name"]
        
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), image)
    print(f"✅ Saved annotated image: {output_path}")
    
    return output_path


def demo_yolo_onnx(image_path: Path, output_dir: Path):
    """Demo YOLO with ONNX Runtime"""
    print("\n🔥 YOLO ONNX Runtime Demo")
    print("-" * 40)
    
    # Create dummy YOLO ONNX model
    model_path = output_dir / "yolov8n.onnx"
    model_path.touch()
    
    print(f"📄 Model: {model_path}")
    print(f"🖼️  Input: {image_path}")
    
    # Show expected usage pattern
    print("""
💡 Real Usage:
   from inferx import InferenceEngine
   
   # YOLO automatically detected by filename
   engine = InferenceEngine("yolov8n.onnx", device="auto")
   result = engine.predict("test_image.jpg")
""")
    
    # Simulate inference
    result = simulate_yolo_inference(image_path, "yolo_onnx")
    
    # Display results
    print(f"📊 Results Summary:")
    print(f"   Detections: {result['num_detections']}")
    print(f"   Inference time: {result['timing']['inference_time']:.3f}s")
    
    for i, detection in enumerate(result["detections"]):
        print(f"   {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
    
    # Draw results
    output_image = output_dir / "yolo_onnx_result.jpg"
    draw_detections_on_image(image_path, result["detections"], output_image)
    
    # Save JSON results
    json_output = output_dir / "yolo_onnx_result.json"
    with open(json_output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"📄 Results saved: {json_output}")
    
    return result


def demo_yolo_openvino(image_path: Path, output_dir: Path):
    """Demo YOLO with OpenVINO Runtime"""
    print("\n🔥 YOLO OpenVINO Runtime Demo")
    print("-" * 40)
    
    # Create dummy YOLO OpenVINO model
    xml_path = output_dir / "yolov8n.xml"
    bin_path = output_dir / "yolov8n.bin"
    xml_path.touch()
    bin_path.touch()
    
    print(f"📄 Model: {xml_path}")
    print(f"🖼️  Input: {image_path}")
    
    # Show expected usage pattern
    print("""
💡 Real Usage:
   from inferx import InferenceEngine
   
   # OpenVINO YOLO with device optimization
   engine = InferenceEngine(
       "yolov8n.xml", 
       device="cpu",         # cpu, gpu, myriad
       runtime="openvino"
   )
   result = engine.predict("test_image.jpg")
""")
    
    # Simulate inference (OpenVINO typically faster)
    result = simulate_yolo_inference(image_path, "yolo_openvino")
    
    # Display results
    print(f"📊 Results Summary:")
    print(f"   Detections: {result['num_detections']}")
    print(f"   Inference time: {result['timing']['inference_time']:.3f}s (OpenVINO optimized)")
    
    for i, detection in enumerate(result["detections"]):
        print(f"   {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
    
    # Draw results
    output_image = output_dir / "yolo_openvino_result.jpg"
    draw_detections_on_image(image_path, result["detections"], output_image)
    
    # Save JSON results
    json_output = output_dir / "yolo_openvino_result.json"
    with open(json_output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"📄 Results saved: {json_output}")
    
    return result


def demo_performance_comparison(onnx_result: dict, openvino_result: dict):
    """Compare ONNX vs OpenVINO performance"""
    print("\n📊 Performance Comparison")
    print("-" * 40)
    
    onnx_time = onnx_result["timing"]["inference_time"]
    openvino_time = openvino_result["timing"]["inference_time"]
    
    speedup = onnx_time / openvino_time
    
    print(f"ONNX Runtime:     {onnx_time:.3f}s")
    print(f"OpenVINO Runtime: {openvino_time:.3f}s")
    print(f"Speedup:          {speedup:.2f}x {'(OpenVINO faster)' if speedup > 1 else '(ONNX faster)'}")
    
    print(f"\n💡 Real-world performance factors:")
    print(f"   - Hardware: OpenVINO optimized for Intel CPUs/GPUs/VPUs")
    print(f"   - Model format: .xml models often more optimized than .onnx")
    print(f"   - Device selection: GPU can be much faster than CPU")
    print(f"   - Batch size: Larger batches usually improve throughput")


def demo_cli_usage(output_dir: Path):
    """Demonstrate CLI usage patterns"""
    print("\n🖥️  CLI Usage Examples")
    print("-" * 40)
    
    model_onnx = output_dir / "yolov8n.onnx"
    model_xml = output_dir / "yolov8n.xml"
    image_path = output_dir / "test_street_scene.jpg"
    
    examples = [
        f"# Basic YOLO detection",
        f"inferx run {model_onnx} {image_path}",
        f"",
        f"# OpenVINO with CPU optimization",
        f"inferx run {model_xml} {image_path} --device cpu --runtime openvino",
        f"",
        f"# GPU acceleration (if available)",
        f"inferx run {model_xml} {image_path} --device gpu",
        f"",
        f"# Intel VPU (Myriad stick)",
        f"inferx run {model_xml} {image_path} --device myriad",
        f"",
        f"# Batch processing with output",
        f"inferx run {model_xml} images/ --output results.json --verbose",
        f"",
        f"# Cross-runtime: ONNX model on OpenVINO",
        f"inferx run {model_onnx} {image_path} --runtime openvino",
        f"",
        f"# With custom configuration",
        f"inferx run {model_xml} {image_path} --config custom_yolo.yaml",
    ]
    
    for example in examples:
        print(f"   {example}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="YOLO Detection Demo with InferX")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--image-size", type=str, default="1280x720", help="Test image size (WxH)")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="inferx_yolo_demo_"))
    
    # Parse image size
    width, height = map(int, args.image_size.split('x'))
    
    print("🎯 InferX YOLO Detection Demo")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Image size: {width}x{height}")
    
    # Create test image
    test_image = create_test_image_with_objects(
        output_dir / "test_street_scene.jpg", 
        size=(width, height)
    )
    
    # Run demos
    try:
        # Demo ONNX YOLO
        onnx_result = demo_yolo_onnx(test_image, output_dir)
        
        # Demo OpenVINO YOLO  
        openvino_result = demo_yolo_openvino(test_image, output_dir)
        
        # Performance comparison
        demo_performance_comparison(onnx_result, openvino_result)
        
        # CLI examples
        demo_cli_usage(output_dir)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("   This is expected when using dummy model files")
    
    # Summary
    print(f"\n🎉 Demo Complete!")
    print(f"   Results saved in: {output_dir}")
    print(f"   ✅ Test image created")
    print(f"   ✅ ONNX inference simulated")
    print(f"   ✅ OpenVINO inference simulated")
    print(f"   ✅ Results annotated and saved")
    
    print(f"\n📁 Generated Files:")
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"   - {file_path.name}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Get real YOLO models (YOLOv8, YOLOv11)")
    print(f"   2. Convert to ONNX: model.export(format='onnx')")  
    print(f"   3. Convert to OpenVINO: mo --input_model model.onnx")
    print(f"   4. Run real inference: inferx run yolov8n.onnx image.jpg")
    

if __name__ == "__main__":
    main()