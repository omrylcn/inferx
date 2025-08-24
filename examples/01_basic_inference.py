#!/usr/bin/env python3
"""
01 - Basic InferX Inference (START HERE!)
==========================================

This is your first InferX example! In 5 minutes you'll learn:
- How to load a YOLO model 
- Run inference on an image
- Compare ONNX vs OpenVINO performance

🎯 GOAL: Get familiar with InferX basics
⏱️  TIME: 5 minutes
📋 YOU'LL BUILD: Working YOLO object detection
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for importing inferx
sys.path.insert(0, str(Path(__file__).parent.parent))

from inferx import InferenceEngine

def main():
    print("🚀 InferX Basic Inference Example")
    print("=" * 50)
    
    print("\n📋 What we'll do:")
    print("1. Load a YOLO model")
    print("2. Run inference on a test image")  
    print("3. See the results!")
    
    # Check if we have test data
    project_root = Path(__file__).parent.parent
    test_image = project_root / "data" / "person.jpeg"
    yolo_onnx = project_root / "models" / "yolo11n_onnx" / "yolo11n.onnx"
    yolo_openvino = project_root / "models" / "yolo11n_openvino" / "yolo11n.xml"
    
    if not test_image.exists():
        print("❌ Test image not found. Please ensure data/person.jpeg exists")
        return
    
    print(f"\n📁 Using test image: {test_image.name}")
    
    # Example 1: ONNX Model
    if yolo_onnx.exists():
        print(f"\n📦 Example 1: ONNX Model ({yolo_onnx.name})")
        print("-" * 30)
        
        try:
            # Load model
            print("⏳ Loading YOLO ONNX model...")
            start_time = time.time()
            engine = InferenceEngine(yolo_onnx, device="auto", runtime="auto")
            load_time = time.time() - start_time
            
            # Run inference
            print("🔍 Running inference...")
            inference_start = time.time()
            result = engine.predict(test_image)
            inference_time = time.time() - inference_start
            
            # Show results
            print("✅ Inference complete!")
            print(f"⏱️  Model load: {load_time:.3f}s")
            print(f"⏱️  Inference:  {inference_time:.3f}s")
            print(f"🎯 Detections: {result.get('num_detections', 0)}")
            
            # Show first detection if any
            detections = result.get('detections', [])
            if detections:
                det = detections[0]
                print(f"🏷️  First detection: {det.get('class_name', 'unknown')} "
                      f"({det.get('confidence', 0):.1%} confidence)")
            
        except Exception as e:
            print(f"❌ ONNX model failed: {e}")
    else:
        print(f"\n⚠️  ONNX model not found at: {yolo_onnx}")
    
    # Example 2: OpenVINO Model  
    if yolo_openvino.exists():
        print(f"\n🧠 Example 2: OpenVINO Model ({yolo_openvino.name})")
        print("-" * 30)
        
        try:
            # Load model
            print("⏳ Loading YOLO OpenVINO model...")
            start_time = time.time()
            engine = InferenceEngine(yolo_openvino, device="auto", runtime="auto")
            load_time = time.time() - start_time
            
            # Run inference  
            print("🔍 Running inference...")
            inference_start = time.time()
            result = engine.predict(test_image)
            inference_time = time.time() - inference_start
            
            # Show results
            print("✅ Inference complete!")
            print(f"⏱️  Model load: {load_time:.3f}s")
            print(f"⏱️  Inference:  {inference_time:.3f}s")
            print(f"🎯 Detections: {result.get('num_detections', 0)}")
            
            # Show first detection if any
            detections = result.get('detections', [])
            if detections:
                det = detections[0]
                print(f"🏷️  First detection: {det.get('class_name', 'unknown')} "
                      f"({det.get('confidence', 0):.1%} confidence)")
                      
        except Exception as e:
            print(f"❌ OpenVINO model failed: {e}")
    else:
        print(f"\n⚠️  OpenVINO model not found at: {yolo_openvino}")
    
    print(f"\n🎉 Basic inference example complete!")
    print(f"🏃 Next: Try '02_batch_processing.py' to process multiple images")
    print(f"📖 Or read USAGE.md for complete feature guide")

if __name__ == "__main__":
    main()