"""Template generator for InferX projects"""

import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from ..settings import validate_yolo_template_config


class TemplateGenerator:
    def __init__(self):
        """Initialize template generator with paths"""
        self.inferx_root = Path(__file__).parent.parent
        self.templates_dir = self.inferx_root / "generators" / "templates"
        self.inferencers_dir = self.inferx_root / "inferencers"
    
    def create_template(self, model_type: str, project_name: str, with_api: bool = False, with_docker: bool = False):
        """Create a new template project"""
        project_path = Path(project_name).resolve()
        
        # Create project directory
        project_path.mkdir(exist_ok=True)
        
        # Copy base template files
        self._copy_base_template(model_type, project_path)
        
        # Copy Docker files if requested
        if with_docker:
            self._copy_docker_files(project_path)
        
        # Copy API files if requested
        if with_api:
            self._copy_api_files(project_path)
        
        # Copy sample data if available
        self._copy_sample_data(project_path)
        
        # Update README with project-specific information
        self._update_readme(project_path, model_type, with_api, with_docker)
        
        print(f"✅ Created {model_type} template: {project_name}")
        print(f"✅ Generated {model_type} project: {project_name}")
        if with_api:
            print("   🌐 API server added")
            print("   To run the server: uv run --extra api python -m src.server")
            print("   API docs available at: http://localhost:8080/docs")
        if with_docker:
            print("   🐳 Docker container added")
        
        return project_path
    
    def _copy_base_template(self, model_type: str, project_path: Path):
        """Copy base template files from templates directory"""
        # Determine template source directory
        if model_type == "yolo" or model_type == "yolo_openvino":
            template_src = self.templates_dir / "yolo"
        else:
            # For other model types, use YOLO template as fallback
            template_src = self.templates_dir / "yolo"
        
        # Copy all files from template directory except src directory and Docker files
        if template_src.exists():
            for item in template_src.iterdir():
                # Skip src directory and Docker files (they will be copied only when needed)
                if item.name in ["src", "Dockerfile", "docker-compose.yml", ".dockerignore"]:
                    continue
                    
                dst_path = project_path / item.name
                if item.is_dir():
                    self._copy_directory(item, dst_path)
                else:
                    # Don't copy __pycache__ directories or .pyc files
                    if "__pycache__" not in str(item) and not item.name.endswith(".pyc"):
                        shutil.copy2(item, dst_path)
        else:
            print(f"Warning: Template directory {template_src} not found")
        
        # Create src directory
        src_path = project_path / "src"
        src_path.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = src_path / "__init__.py"
        init_file.write_text("")
        
        # Create models directory
        models_path = project_path / "models"
        models_path.mkdir(exist_ok=True)
        (models_path / ".gitkeep").write_text("# Keep this directory in git")
        
        # Update config.yaml with model-specific settings
        self._update_config_for_model_type(model_type, project_path)
        
        # Add OpenVINO dependencies if needed
        if model_type == "yolo_openvino":
            self._update_pyproject_for_openvino(project_path)
        
        # Create inferencer.py based on model type
        self._create_inferencer_py(model_type, src_path)
    
    def _copy_directory(self, src: Path, dst: Path):
        """Copy directory recursively, excluding __pycache__ and .pyc files"""
        dst.mkdir(exist_ok=True)
        for item in src.iterdir():
            dst_path = dst / item.name
            if item.is_dir():
                if "__pycache__" not in str(item):
                    self._copy_directory(item, dst_path)
            else:
                if not item.name.endswith(".pyc") and "__pycache__" not in str(item):
                    shutil.copy2(item, dst_path)
    
    def _update_config_for_model_type(self, model_type: str, project_path: Path):
        """Update config.yaml with model-specific settings"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            return
        
        config_content = config_file.read_text()
        
        # Update model path and type based on model_type
        if model_type == "yolo_openvino":
            config_content = config_content.replace(
                'path: "models/yolo_model.onnx"', 
                'path: "models/yolo_model.xml"'
            )
            config_content = config_content.replace(
                'type: "yolo"', 
                'type: "yolo_openvino"'
            )
            # Update runtime if it exists
            if 'runtime: "auto"' in config_content:
                config_content = config_content.replace(
                    'runtime: "auto"', 
                    'runtime: "openvino"'
                )
            # Create placeholder files for OpenVINO
            models_path = project_path / "models"
            (models_path / "yolo_model.xml").write_text("# Place your YOLO OpenVINO model .xml file here")
            (models_path / "yolo_model.bin").write_text("# Place your YOLO OpenVINO model .bin file here")
        elif model_type == "yolo":
            # Already correct for YOLO ONNX, just create placeholder
            models_path = project_path / "models"
            (models_path / "yolo_model.onnx").write_text("# Place your YOLO ONNX model file here")
        else:
            # Generic model
            config_content = config_content.replace(
                'path: "models/yolo_model.onnx"', 
                'path: "models/model.onnx"'
            )
            config_content = config_content.replace(
                'type: "yolo"', 
                'type: "generic"'
            )
            # Create placeholder for generic model
            models_path = project_path / "models"
            (models_path / "model.onnx").write_text("# Place your model file here")
        
        config_file.write_text(config_content)
    
    def _update_pyproject_for_openvino(self, project_path: Path):
        """Update pyproject.toml to include OpenVINO dependencies"""
        pyproject_file = project_path / "pyproject.toml"
        if not pyproject_file.exists():
            return
        
        content = pyproject_file.read_text()
        # Add OpenVINO to main dependencies
        if "openvino>=2023.3.0" not in content:
            # Find the dependencies section and add OpenVINO
            content = content.replace(
                'dependencies = [',
                'dependencies = [\n    "openvino>=2023.3.0",'
            )
        pyproject_file.write_text(content)
    
    def _create_inferencer_py(self, model_type: str, src_path: Path):
        """Create self-contained inferencer.py with all dependencies"""
        if model_type == "yolo_openvino":
            self._create_yolo_openvino_inferencer(src_path)
        elif model_type == "yolo":
            self._create_yolo_inferencer(src_path)
        else:
            # Generic inferencer for other model types
            self._create_generic_inferencer(src_path)
    
    def _create_yolo_inferencer(self, src_path: Path):
        """Create YOLO inferencer with all dependencies"""
        # Copy yolo.py and fix import paths
        yolo_source = self.inferx_root / "inferencers" / "yolo.py"
        if yolo_source.exists():
            content = yolo_source.read_text()
            # Fix import paths to relative imports
            content = content.replace("from .base import BaseInferencer", 
                                    "from .base import BaseInferencer")
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from .utils import ImageProcessor")
            content = content.replace("from .yolo_base import BaseYOLOInferencer", 
                                    "from .yolo_base import BaseYOLOInferencer")
            content = content.replace("from ..exceptions import (", 
                                    "from .exceptions import (")
            # Rename class to Inferencer
            content = content.replace("class YOLOInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("base.py", src_path)
            self._copy_dependency("utils.py", src_path)
            self._copy_dependency("exceptions.py", src_path)
            self._copy_dependency("yolo_base.py", src_path)
    
    def _create_yolo_openvino_inferencer(self, src_path: Path):
        """Create YOLO OpenVINO inferencer with all dependencies"""
        # Copy yolo_openvino.py and fix import paths
        yolo_source = self.inferx_root / "inferencers" / "yolo_openvino.py"
        if yolo_source.exists():
            content = yolo_source.read_text()
            # Fix import paths to relative imports
            content = content.replace("from .base import BaseInferencer", 
                                    "from .base import BaseInferencer")
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from .utils import ImageProcessor")
            content = content.replace("from ..exceptions import (", 
                                    "from .exceptions import (")
            content = content.replace("from ..exceptions import ModelError, ErrorCode", 
                                    "from .exceptions import ModelError, ErrorCode")
            # Rename class to Inferencer
            content = content.replace("class YOLOOpenVINOInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("base.py", src_path)
            self._copy_dependency("utils.py", src_path)
            self._copy_dependency("exceptions.py", src_path)
            self._copy_dependency("yolo_base.py", src_path)
    
    def _create_generic_inferencer(self, src_path: Path):
        """Create generic inferencer"""
        # Copy base.py and fix import paths
        base_source = self.inferx_root / "inferencers" / "base.py"
        if base_source.exists():
            content = base_source.read_text()
            # Fix import paths to relative imports
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from .utils import ImageProcessor")
            content = content.replace("from ..exceptions import (", 
                                    "from .exceptions import (")
            # Rename class to Inferencer
            content = content.replace("class BaseInferencer", "class Inferencer")
            (src_path / "inferencer.py").write_text(content)
            
            # Copy dependencies
            self._copy_dependency("utils.py", src_path)
            self._copy_dependency("exceptions.py", src_path)
    
    def _copy_dependency(self, filename: str, src_path: Path):
        """Copy a dependency file from inferx to src directory"""
        # First try inferencers directory, then fall back to root inferx directory
        source_file = self.inferx_root / "inferencers" / filename
        if not source_file.exists():
            source_file = self.inferx_root / filename
            
        if source_file.exists():
            content = source_file.read_text()
            # Fix import paths to relative imports
            content = content.replace("from ..utils import ImageProcessor", 
                                    "from .utils import ImageProcessor")
            content = content.replace("from ..exceptions import (", 
                                    "from .exceptions import (")
            content = content.replace("from .base import BaseInferencer", 
                                    "from .base import BaseInferencer")
            content = content.replace("from ..exceptions import ModelError, ErrorCode", 
                                    "from .exceptions import ModelError, ErrorCode")
            (src_path / filename).write_text(content)
        else:
            print(f"DEBUG: Source file {source_file} not found")
    
    def _copy_api_files(self, project_path: Path):
        """Copy FastAPI server files to project"""
        src_path = project_path / "src"
        
        # Copy server.py from template
        template_server = self.templates_dir / "yolo" / "src" / "server.py"
        if template_server.exists():
            dst_server = src_path / "server.py"
            import shutil
            shutil.copy2(template_server, dst_server)
            
            # Update pyproject.toml to include API dependencies
            self._update_pyproject_for_api(project_path)
    
    def _copy_docker_files(self, project_path: Path):
        """Copy Docker files from template"""
        # Copy Dockerfile
        dockerfile_src = self.templates_dir / "yolo" / "Dockerfile"
        if dockerfile_src.exists():
            shutil.copy2(dockerfile_src, project_path / "Dockerfile")
        
        # Copy docker-compose.yml
        compose_src = self.templates_dir / "yolo" / "docker-compose.yml"
        if compose_src.exists():
            shutil.copy2(compose_src, project_path / "docker-compose.yml")
        
        # Copy .dockerignore
        dockerignore_src = self.templates_dir / "yolo" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy2(dockerignore_src, project_path / ".dockerignore")
    
    def _update_config(self, project_path: Path, options: dict):
        """Update config file with user options"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            return
        
        # Read current config
        config_content = config_file.read_text()
        
        # Update device if specified
        if "device" in options and options["device"] != "auto":
            config_content = config_content.replace(
                'device: "auto"', 
                f'device: "{options["device"]}"'
            )
        
        # Update runtime if specified
        if "runtime" in options and options["runtime"] != "auto":
            # Add runtime section if it doesn't exist
            if "runtime:" not in config_content:
                config_content = config_content.replace(
                    "inference:",
                    'inference:\\n  runtime: "auto"'
                )
            config_content = config_content.replace(
                'runtime: "auto"',
                f'runtime: "{options["runtime"]}"'
            )
        
        # Write updated config
        config_file.write_text(config_content)
    
    def _update_pyproject_for_api(self, project_path: Path):
        """Update pyproject.toml to include API dependencies"""
        pyproject_file = project_path / "pyproject.toml"
        if not pyproject_file.exists():
            return
        
        content = pyproject_file.read_text()
        # Add API extra if not exists
        if "[project.optional-dependencies]" not in content:
            content += '''
[project.optional-dependencies]
api = [
    "fastapi>=0.68.0",
    "uvicorn>=0.15.0",
]
'''
        elif "api = [" not in content:
            # Find the end of optional-dependencies section and add api
            lines = content.splitlines()
            new_lines = []
            in_optional = False
            optional_ended = False
            
            for line in lines:
                new_lines.append(line)
                if line.strip() == "[project.optional-dependencies]":
                    in_optional = True
                elif in_optional and line.startswith("[") and line.strip() != "[project.optional-dependencies]":
                    # Add api before next section
                    new_lines.insert(-1, 'api = [')
                    new_lines.insert(-1, '    "fastapi>=0.68.0",')
                    new_lines.insert(-1, '    "uvicorn>=0.15.0",')
                    new_lines.insert(-1, ']')
                    in_optional = False
                    optional_ended = True
            
            if in_optional and not optional_ended:
                # Add at the end
                new_lines.append('api = [')
                new_lines.append('    "fastapi>=0.68.0",')
                new_lines.append('    "uvicorn>=0.15.0",')
                new_lines.append(']')
            
            content = "\n".join(new_lines)
        
        pyproject_file.write_text(content)
    
    def _validate_generated_config(self, project_path: Path):
        """Validate generated template configuration with Pydantic"""
        config_file = project_path / "config.yaml"
        if not config_file.exists():
            print("⚠️ Warning: config.yaml not found, skipping validation")
            return
        
        try:
            validated_config = validate_yolo_template_config(config_file)
            print("✅ YOLO template configuration validated with Pydantic")
            print(f"   Model: {validated_config.model_path}")
            print(f"   Input size: {validated_config.input_size}")
            print(f"   Confidence: {validated_config.confidence_threshold}")
        except ImportError:
            print("⚠️ pydantic-settings not available, skipping validation")
        except Exception as e:
            print(f"⚠️ Configuration validation warning: {e}")
            print("   Template created but config may need adjustment")
    
    def _copy_sample_data(self, project_path: Path):
        """Copy sample data files from data directory"""
        data_src = self.inferx_root / "data"
        data_dst = project_path / "data"
        
        if data_src.exists():
            data_dst.mkdir(exist_ok=True)
            # Copy sample images
            for item in data_src.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dst_file = data_dst / item.name
                    shutil.copy2(item, dst_file)
    
    def _update_readme(self, project_path: Path, model_type: str, with_api: bool, with_docker: bool):
        """Update README.md with project-specific information"""
        readme_file = project_path / "README.md"
        if not readme_file.exists():
            return
            
        content = readme_file.read_text()
        
        # Update title
        content = content.replace(
            "# InferX Template Project", 
            f"# {model_type.upper()} InferX Project"
        )
        
        # Add information about optional features
        if with_api or with_docker:
            features_section = "\n## 🚀 Project Features\n\n"
            if with_api:
                features_section += "- **FastAPI Server** - REST API for inference\n"
            if with_docker:
                features_section += "- **Docker Container** - Containerized deployment\n"
            
            # Insert after the first paragraph
            lines = content.split('\n')
            if len(lines) > 1:
                lines.insert(2, features_section)
                content = '\n'.join(lines)
        
        readme_file.write_text(content)
    
    def copy_model_to_template(self, model_path: str, project_path: Path, model_type: str):
        """Copy model file to template project"""
        model_src = Path(model_path)
        models_dir = project_path / "models"
        
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type == "yolo_openvino":
            # For OpenVINO, we might have multiple files
            if model_src.is_dir():
                # Copy all files from directory
                for item in model_src.iterdir():
                    if item.suffix in [".xml", ".bin"]:
                        dst_file = models_dir / f"yolo_model{item.suffix}"
                        shutil.copy2(item, dst_file)
            else:
                # Assume it's XML file, look for BIN file
                if model_src.suffix == ".xml":
                    dst_xml = models_dir / "yolo_model.xml"
                    shutil.copy2(model_src, dst_xml)
                    
                    # Look for corresponding BIN file
                    bin_file = model_src.with_suffix(".bin")
                    if bin_file.exists():
                        dst_bin = models_dir / "yolo_model.bin"
                        shutil.copy2(bin_file, dst_bin)
        else:
            # For ONNX and other models
            dst_file = models_dir / f"yolo_model{model_src.suffix}"
            shutil.copy2(model_src, dst_file)
    
    def add_api_layer(self, project_path: str):
        """Add FastAPI server layer to existing project"""
        project_path = Path(project_path).resolve()
        
        # Copy server.py from template
        self._copy_api_files(project_path)
        
        # Update pyproject.toml to include API dependencies
        self._update_pyproject_for_api(project_path)
        
        print("✅ Added FastAPI server to project")
        print("   To run the server: uv run --extra api python -m src.server")
        print("   API docs available at: http://localhost:8080/docs")
    
    def _copy_api_files(self, project_path: Path):
        """Copy FastAPI server files to project"""
        src_path = project_path / "src"
        
        # Copy server.py from template
        template_server = self.templates_dir / "yolo" / "src" / "server.py"
        if template_server.exists():
            dst_server = src_path / "server.py"
            import shutil
            shutil.copy2(template_server, dst_server)
            
            # Update pyproject.toml to include API dependencies
            self._update_pyproject_for_api(project_path)