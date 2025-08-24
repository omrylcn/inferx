#!/usr/bin/env python3
"""
03 - CLI Examples with InferX
=============================

See InferX CLI in action with interactive demonstrations!

🎯 GOAL: Master the InferX command line interface  
⏱️  TIME: 3 minutes
📋 YOU'LL BUILD: CLI workflow skills
"""

import sys
import subprocess
from pathlib import Path
import time
import glob

def run_command(cmd, description):
    """Run a command and show the output"""
    print(f"\n🖥️  {description}")
    print("─" * len(f"🖥️  {description}"))
    print(f"💻 Command: {cmd}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("❌ Error:")
            print(result.stderr)
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (30s)")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def find_test_resources():
    """Find available models and images"""
    project_root = Path(__file__).parent.parent
    
    # Find models
    models_dir = project_root / "models"
    yolo_onnx = models_dir / "yolo11n_onnx" / "yolo11n.onnx"
    yolo_openvino = models_dir / "yolo11n_openvino" / "yolo11n.xml"
    
    available_models = []
    if yolo_onnx.exists():
        available_models.append(("ONNX", str(yolo_onnx)))
    if yolo_openvino.exists():
        available_models.append(("OpenVINO", str(yolo_openvino)))
    
    # Find images
    data_dir = project_root / "data"
    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    available_images = []
    
    for pattern in image_patterns:
        available_images.extend(glob.glob(str(data_dir / pattern)))
    
    return available_models, available_images

def main():
    print("🚀 InferX CLI Examples")
    print("=" * 50)
    
    print("\n📋 What we'll demonstrate:")
    print("1. Basic CLI inference")
    print("2. Device selection options")
    print("3. Output formatting")
    print("4. Batch processing with CLI")
    
    # Check for test resources
    models, images = find_test_resources()
    
    if not models:
        print("\n❌ No models found! Please ensure you have:")
        print("   - models/yolo11n_onnx/yolo11n.onnx")
        print("   - models/yolo11n_openvino/yolo11n.xml")
        return
    
    if not images:
        print("\n❌ No test images found! Please add images to data/ directory")
        return
    
    print(f"\n📦 Available models: {len(models)}")
    for model_type, model_path in models:
        print(f"   - {model_type}: {Path(model_path).name}")
    
    print(f"\n📷 Available images: {len(images)}")
    for image in images[:3]:  # Show first 3
        print(f"   - {Path(image).name}")
    if len(images) > 3:
        print(f"   - ... and {len(images)-3} more")
    
    # Use first available model and image
    model_type, model_path = models[0]
    test_image = images[0]
    
    print(f"\n🎯 Using {model_type} model with {Path(test_image).name}")
    
    # Example 1: Basic CLI inference
    print("\n" + "="*60)
    print("📚 EXAMPLE 1: Basic CLI Inference")
    print("="*60)
    
    cmd = f"uv run inferx run \"{model_path}\" \"{test_image}\""
    success = run_command(cmd, "Basic inference with auto-detection")
    
    if not success:
        print("⚠️  CLI might need setup. Continuing with other examples...")
    
    # Example 2: Device selection
    print("\n" + "="*60)
    print("📚 EXAMPLE 2: Device Selection")
    print("="*60)
    
    device_examples = [
        ("CPU inference", f"uv run inferx run \"{model_path}\" \"{test_image}\" --device cpu"),
        ("Auto device selection", f"uv run inferx run \"{model_path}\" \"{test_image}\" --device auto"),
    ]
    
    # Add GPU example if it's an appropriate model
    if "openvino" in model_path.lower():
        device_examples.append(
            ("Intel GPU (if available)", f"uv run inferx run \"{model_path}\" \"{test_image}\" --device gpu")
        )
    
    for desc, cmd in device_examples:
        run_command(cmd, desc)
    
    # Example 3: Output formatting
    print("\n" + "="*60)
    print("📚 EXAMPLE 3: Output Options")
    print("="*60)
    
    output_examples = [
        ("Verbose output", f"uv run inferx run \"{model_path}\" \"{test_image}\" --verbose"),
        ("JSON output", f"uv run inferx run \"{model_path}\" \"{test_image}\" --output results.json"),
    ]
    
    for desc, cmd in output_examples:
        run_command(cmd, desc)
        
        # Check if JSON file was created
        if "results.json" in cmd:
            json_file = Path("results.json")
            if json_file.exists():
                print(f"📄 Results saved to: {json_file}")
                try:
                    content = json_file.read_text()[:200] + "..." if len(json_file.read_text()) > 200 else json_file.read_text()
                    print(f"📝 Content preview:\n{content}")
                except:
                    print("📄 JSON file created successfully")
    
    # Example 4: Runtime selection  
    print("\n" + "="*60)
    print("📚 EXAMPLE 4: Runtime Selection")
    print("="*60)
    
    runtime_examples = [
        ("Auto runtime", f"uv run inferx run \"{model_path}\" \"{test_image}\" --runtime auto"),
    ]
    
    # Add specific runtime examples based on available models
    for model_type, model_path_iter in models:
        if model_type == "ONNX":
            runtime_examples.append(
                ("Force ONNX runtime", f"uv run inferx run \"{model_path_iter}\" \"{test_image}\" --runtime onnx")
            )
        elif model_type == "OpenVINO":
            runtime_examples.append(
                ("Force OpenVINO runtime", f"uv run inferx run \"{model_path_iter}\" \"{test_image}\" --runtime openvino")
            )
    
    for desc, cmd in runtime_examples:
        run_command(cmd, desc)
    
    # Example 5: Batch processing
    if len(images) > 1:
        print("\n" + "="*60)
        print("📚 EXAMPLE 5: Batch Processing")
        print("="*60)
        
        data_dir = Path(images[0]).parent
        
        batch_examples = [
            ("Process directory", f"uv run inferx run \"{model_path}\" \"{data_dir}\" --output batch_results.json"),
            ("Process with verbose output", f"uv run inferx run \"{model_path}\" \"{data_dir}\" --verbose"),
        ]
        
        for desc, cmd in batch_examples:
            run_command(cmd, desc)
    
    # Example 6: Help and info
    print("\n" + "="*60)
    print("📚 EXAMPLE 6: Help and Information")
    print("="*60)
    
    help_examples = [
        ("InferX help", "uv run inferx --help"),
        ("Run command help", "uv run inferx run --help"),
        ("Version info", "uv run inferx --version"),
    ]
    
    for desc, cmd in help_examples:
        run_command(cmd, desc)
    
    # Summary
    print("\n" + "="*60)
    print("🎉 CLI Examples Complete!")
    print("="*60)
    
    print("\n📚 What you learned:")
    print("✅ Basic inference: inferx run model.onnx image.jpg")
    print("✅ Device selection: --device cpu/gpu/auto")
    print("✅ Runtime control: --runtime onnx/openvino/auto")
    print("✅ Output options: --output results.json --verbose")
    print("✅ Batch processing: inferx run model.onnx images/")
    
    print("\n💡 Pro CLI Tips:")
    print("1. Use --verbose for debugging")
    print("2. Save results with --output for analysis")
    print("3. Batch process directories for efficiency")
    print("4. Auto-detection works great for most cases")
    print("5. Combine options: --device gpu --runtime openvino --verbose")
    
    print("\n🔧 Common CLI Patterns:")
    print("# Quick inference")
    print("inferx run model.onnx image.jpg")
    print()
    print("# Production inference with logging")
    print("inferx run model.xml images/ --output results.json --device gpu --verbose")
    print()
    print("# Performance testing")
    print("inferx run model.onnx image.jpg --device cpu --verbose")
    print("inferx run model.xml image.jpg --device gpu --verbose")
    
    print(f"\n🏃 Next: Try '04_template_walkthrough.py' to create full projects!")

if __name__ == "__main__":
    main()