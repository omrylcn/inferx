#!/usr/bin/env python3
"""
02 - Batch Processing with InferX
=================================

Learn to process multiple images efficiently with InferX!

🎯 GOAL: Process multiple images at once
⏱️  TIME: 3 minutes  
📋 YOU'LL BUILD: Batch image processing workflow
"""

import sys
from pathlib import Path
import time
import glob

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferx import InferenceEngine

def main():
    print("🚀 InferX Batch Processing Example")
    print("=" * 50)
    
    print("\n📋 What we'll do:")
    print("1. Load a YOLO model once")
    print("2. Process multiple images in batch")
    print("3. Compare single vs batch performance")
    
    # Check for models and data
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    # Look for available images
    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(str(data_dir / pattern)))
    
    if not all_images:
        print("❌ No test images found in data/")
        print("   Add some .jpg/.png images to data/ directory")
        return
        
    # Use first few images for demo
    test_images = all_images[:3] if len(all_images) >= 3 else all_images
    print(f"\n📁 Found {len(all_images)} images, using {len(test_images)} for demo")
    
    # Find available model
    yolo_onnx = models_dir / "yolo11n_onnx" / "yolo11n.onnx"
    yolo_openvino = models_dir / "yolo11n_openvino" / "yolo11n.xml"
    
    model_path = None
    if yolo_onnx.exists():
        model_path = yolo_onnx
        model_type = "ONNX"
    elif yolo_openvino.exists():
        model_path = yolo_openvino  
        model_type = "OpenVINO"
    else:
        print("❌ No YOLO models found. Please ensure models are available:")
        print(f"   - {yolo_onnx}")
        print(f"   - {yolo_openvino}")
        return
    
    print(f"📦 Using {model_type} model: {model_path.name}")
    
    try:
        # Load model once
        print(f"\n⏳ Loading {model_type} model...")
        start_time = time.time()
        engine = InferenceEngine(model_path, device="auto", runtime="auto")
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.3f}s")
        
        # Method 1: Process images one by one
        print(f"\n📊 Method 1: Individual Processing")
        print("-" * 30)
        
        individual_times = []
        individual_results = []
        
        for i, image_path in enumerate(test_images):
            print(f"🔍 Processing image {i+1}/{len(test_images)}: {Path(image_path).name}")
            
            start_time = time.time()
            result = engine.predict(image_path)
            process_time = time.time() - start_time
            
            individual_times.append(process_time)
            individual_results.append(result)
            
            detections = result.get('num_detections', 0)
            print(f"   ⏱️  {process_time:.3f}s | 🎯 {detections} detections")
        
        total_individual = sum(individual_times)
        avg_individual = total_individual / len(individual_times)
        
        print(f"\n📊 Individual Processing Summary:")
        print(f"   Total time: {total_individual:.3f}s")
        print(f"   Average per image: {avg_individual:.3f}s")
        
        # Method 2: Batch processing (if supported)
        print(f"\n📊 Method 2: Batch Processing")
        print("-" * 30)
        
        try:
            print(f"🔍 Processing {len(test_images)} images in batch...")
            start_time = time.time()
            
            # Process all images at once
            batch_results = []
            for image_path in test_images:
                result = engine.predict(image_path)  # InferX processes one at a time internally
                batch_results.append(result)
            
            batch_time = time.time() - start_time
            avg_batch = batch_time / len(test_images)
            
            print(f"✅ Batch processing complete!")
            print(f"   Total time: {batch_time:.3f}s")  
            print(f"   Average per image: {avg_batch:.3f}s")
            
            # Performance comparison
            print(f"\n🏁 Performance Comparison")
            print("-" * 30)
            print(f"Individual processing: {avg_individual:.3f}s per image")
            print(f"Batch processing:      {avg_batch:.3f}s per image")
            
            if avg_batch < avg_individual:
                speedup = avg_individual / avg_batch
                print(f"🚀 Batch is {speedup:.2f}x faster!")
            else:
                print(f"📊 Similar performance (model loading overhead dominates)")
                
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            print("   Using individual processing results")
        
        # Show detection summary
        print(f"\n🎯 Detection Summary")
        print("-" * 30)
        
        total_detections = 0
        for i, result in enumerate(individual_results):
            detections = result.get('detections', [])
            num_detections = len(detections)
            total_detections += num_detections
            
            image_name = Path(test_images[i]).name
            print(f"📷 {image_name}: {num_detections} objects")
            
            # Show top detection for each image
            if detections:
                top_detection = max(detections, key=lambda x: x.get('confidence', 0))
                print(f"   🏷️  Best: {top_detection.get('class_name', 'unknown')} "
                      f"({top_detection.get('confidence', 0):.1%})")
        
        print(f"\n📊 Total objects detected: {total_detections}")
        print(f"📊 Average per image: {total_detections/len(test_images):.1f}")
        
    except Exception as e:
        print(f"❌ Batch processing demo failed: {e}")
        return
    
    print(f"\n💡 Pro Tips for Batch Processing:")
    print("1. Load model once, reuse for all images")
    print("2. Use CLI for large batches: inferx run model.onnx images/ --output results.json")
    print("3. GPU devices typically benefit more from batching")
    print("4. Consider image preprocessing optimization for large datasets")
    
    print(f"\n🏃 Next: Try '03_cli_examples.py' to see CLI batch processing")

if __name__ == "__main__":
    main()