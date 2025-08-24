#!/usr/bin/env python3
"""
04 - Template Generation Walkthrough
====================================

Generate complete InferX projects with full API servers!

🎯 GOAL: Create deployable inference projects
⏱️  TIME: 5 minutes
📋 YOU'LL BUILD: Complete inference API project
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

def run_command(cmd, description, cwd=None):
    """Run a command and show the output"""
    print(f"\n🖥️  {description}")
    print("─" * len(f"🖥️  {description}"))
    print(f"💻 Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=cwd
        )
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("❌ Error:")
            print(result.stderr)
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (60s)")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def generate_template(model_type, name, with_api, output_dir):
    """Generate a template project"""
    print(f"\n🏗️  Generating {name} template...")
    
    cmd = f"uv run inferx template --model-type {model_type} --name {name}"
    if with_api:
        cmd += " --with-api"
    
    success = run_command(cmd, f"Generating {name} ({model_type}{'+ API' if with_api else ''})", cwd=output_dir)
    
    if success:
        project_path = output_dir / name
        if project_path.exists():
            print(f"✅ Template created: {project_path}")
            return project_path
        else:
            print(f"⚠️  Project directory not found: {project_path}")
            return None
    else:
        print(f"❌ Template generation failed")
        return None

def setup_project(project_path):
    """Set up the generated project with UV"""
    print(f"\n⚙️  Setting up project with UV...")
    
    if not project_path or not project_path.exists():
        print("❌ Project path doesn't exist")
        return False
    
    # Install dependencies with UV
    success = run_command("uv sync", f"Installing dependencies for {project_path.name}", cwd=project_path)
    
    if success:
        print(f"✅ Project setup complete: {project_path.name}")
        return True
    else:
        print(f"❌ Project setup failed")
        return False

def test_basic_project(project_path):
    """Test basic functionality of a project"""
    print(f"\n🧪 Testing basic functionality...")
    
    if not project_path or not project_path.exists():
        return False
    
    # Check project structure
    key_files = ["pyproject.toml", "src/inferencer.py"]
    missing_files = []
    
    for file_path in key_files:
        full_path = project_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    else:
        print("✅ Project structure looks good")
    
    # Try to run basic inference
    test_cmd = "uv run python -c \"from src.inferencer import Inferencer; print('✅ Import successful')\""
    success = run_command(test_cmd, "Testing import", cwd=project_path)
    
    return success

def test_api_project(project_path):
    """Test API functionality of a project"""
    print(f"\n🌐 Testing API functionality...")
    
    if not project_path or not project_path.exists():
        return False
    
    # Check for API files
    api_files = ["src/server.py", "src/app.py"]
    has_api = any((project_path / file_path).exists() for file_path in api_files)
    
    if not has_api:
        print("ℹ️  No API files found (basic template)")
        return False
    
    # Test API import
    test_cmd = "uv run python -c \"from src.server import app; print('✅ API import successful')\""
    success = run_command(test_cmd, "Testing API import", cwd=project_path)
    
    if success:
        print("💡 To start API server:")
        print(f"   cd {project_path.name}")
        print("   uv run python src/server.py")
    
    return success

def main():
    print("🚀 InferX Template Generation Walkthrough")
    print("=" * 60)
    
    print("\n📋 What we'll do:")
    print("1. Generate all 4 template types")
    print("2. Set up projects with UV") 
    print("3. Test basic functionality")
    print("4. Test API servers")
    print("5. Show deployment patterns")
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory(prefix="inferx_templates_") as temp_dir:
        workspace = Path(temp_dir)
        print(f"\n📁 Working in: {workspace}")
        
        # Define all 4 template combinations
        templates = [
            ("yolo", "yolo_basic", False, "YOLO ONNX Basic"),
            ("yolo", "yolo_with_api", True, "YOLO ONNX + API"),
            ("yolo_openvino", "yolo_openvino_basic", False, "YOLO OpenVINO Basic"),
            ("yolo_openvino", "yolo_openvino_with_api", True, "YOLO OpenVINO + API")
        ]
        
        generated_projects = []
        
        # Generate all templates
        print("\n" + "="*60)
        print("🏗️  PHASE 1: Template Generation")
        print("="*60)
        
        for model_type, name, with_api, description in templates:
            print(f"\n🎯 Generating: {description}")
            project_path = generate_template(model_type, name, with_api, workspace)
            
            if project_path:
                generated_projects.append((project_path, with_api, description))
            else:
                print(f"⚠️  Skipping {name} due to generation failure")
        
        if not generated_projects:
            print("\n❌ No templates generated successfully")
            return
        
        # Set up projects
        print("\n" + "="*60)
        print("⚙️  PHASE 2: Project Setup")
        print("="*60)
        
        setup_projects = []
        
        for project_path, with_api, description in generated_projects:
            print(f"\n🔧 Setting up: {description}")
            success = setup_project(project_path)
            
            if success:
                setup_projects.append((project_path, with_api, description))
            else:
                print(f"⚠️  Setup failed for {project_path.name}")
        
        # Test projects
        print("\n" + "="*60)
        print("🧪 PHASE 3: Testing Projects")
        print("="*60)
        
        working_basic = []
        working_api = []
        
        for project_path, with_api, description in setup_projects:
            print(f"\n🔬 Testing: {description}")
            
            # Test basic functionality
            basic_works = test_basic_project(project_path)
            
            if basic_works:
                working_basic.append((project_path, description))
                
                # Test API if applicable
                if with_api:
                    api_works = test_api_project(project_path)
                    if api_works:
                        working_api.append((project_path, description))
        
        # Results summary
        print("\n" + "="*60)
        print("🎉 RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Template Generation Results:")
        print(f"   Generated: {len(generated_projects)}/4 templates")
        print(f"   Setup successful: {len(setup_projects)}/4 templates")
        print(f"   Basic functionality: {len(working_basic)} projects")
        print(f"   API functionality: {len(working_api)} projects")
        
        if working_basic:
            print(f"\n✅ Working Basic Projects:")
            for project_path, description in working_basic:
                print(f"   - {description}: {project_path.name}")
        
        if working_api:
            print(f"\n🌐 Working API Projects:")
            for project_path, description in working_api:
                print(f"   - {description}: {project_path.name}")
        
        # Show project structure for the first working project
        if working_basic:
            sample_project = working_basic[0][0]
            print(f"\n📂 Sample Project Structure ({sample_project.name}):")
            print("─" * 40)
            
            def show_tree(path, prefix="", max_depth=2, current_depth=0):
                if current_depth >= max_depth:
                    return
                
                items = sorted(path.iterdir())
                dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
                files = [item for item in items if item.is_file() and not item.name.startswith('.')]
                
                # Show directories first
                for i, item in enumerate(dirs):
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    print(f"{prefix}{'└── ' if is_last_dir else '├── '}{item.name}/")
                    extension = "    " if is_last_dir else "│   "
                    show_tree(item, prefix + extension, max_depth, current_depth + 1)
                
                # Show files
                for i, item in enumerate(files):
                    is_last = (i == len(files) - 1)
                    print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
            
            show_tree(sample_project)
        
        # Deployment examples
        print("\n" + "="*60)
        print("🚀 DEPLOYMENT EXAMPLES")
        print("="*60)
        
        print("\n💻 Local Development:")
        print("# 1. Generate project")
        print("inferx template --model-type yolo --name my_project --with-api")
        print()
        print("# 2. Setup environment")
        print("cd my_project")
        print("uv sync")
        print()
        print("# 3. Add your YOLO model")
        print("mkdir -p models")
        print("cp /path/to/your/yolo.onnx models/")
        print()
        print("# 4. Run API server")
        print("uv run python src/server.py")
        print("# Server will be at: http://localhost:8000")
        
        print("\n🐳 Production Deployment:")
        print("# Templates include everything needed for production:")
        print("# - Proper dependency management (pyproject.toml)")
        print("# - FastAPI server with health checks")
        print("# - Error handling and logging")
        print("# - JSON API responses")
        print("# - UV for fast, reliable builds")
        
        print("\n📚 Template Types Guide:")
        print("┌─────────────────────┬──────────────┬─────────────────────────┐")
        print("│ Template            │ Runtime      │ Best For                │")
        print("├─────────────────────┼──────────────┼─────────────────────────┤")
        print("│ yolo_basic          │ ONNX         │ Simple scripts, testing │")
        print("│ yolo_with_api       │ ONNX         │ Web services, general   │")
        print("│ yolo_openvino_basic │ OpenVINO     │ Intel hardware          │")
        print("│ yolo_openvino_api   │ OpenVINO     │ High-performance APIs   │")
        print("└─────────────────────┴──────────────┴─────────────────────────┘")
    
    print(f"\n🎉 Template Walkthrough Complete!")
    print("You now know how to:")
    print("✅ Generate InferX projects")
    print("✅ Set up development environments")
    print("✅ Test basic and API functionality")
    print("✅ Deploy inference services")
    
    print(f"\n🏃 Ready to build your own inference applications!")

if __name__ == "__main__":
    main()