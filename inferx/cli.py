"""CLI interface for InferX"""

import click
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from . import __version__
from .runtime import InferenceEngine
from .utils import FileUtils
from .settings import get_inferx_settings


@click.group()
@click.version_option(version=__version__, prog_name="InferX")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """InferX - Lightweight ML Inference Runtime
    
    Train how you want. Export how you want. Run with InferX.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not config_path:
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        click.echo(f"❌ Failed to load config {config_path}: {e}", err=True)
        return {}


def _save_results(results: Dict[str, Any], output_path: Path, format: str = "json") -> None:
    """Save inference results to file"""
    try:
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        click.echo(f"✅ Results saved to: {output_path}")
    except Exception as e:
        click.echo(f"❌ Failed to save results: {e}", err=True)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), 
              help="Configuration file path")
@click.option("--output", "-o", type=click.Path(path_type=Path), 
              help="Output file path")
@click.option("--batch-size", type=int, default=1, help="Batch size for inference")
@click.option("--device", type=click.Choice(["auto", "cpu", "gpu"]), default="auto",
              help="Device to run inference on")
@click.option("--runtime", type=click.Choice(["auto", "onnx", "openvino"]), default="auto",
              help="Runtime engine to use")
@click.option("--format", type=click.Choice(["json", "yaml"]), default="json",
              help="Output format")
@click.pass_context
def run(ctx: click.Context, model_path: Path, input_path: Path, 
        config: Optional[Path], output: Optional[Path], batch_size: int,
        device: str, runtime: str, format: str) -> None:
    """Run inference on model with given input"""
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Load configuration
        user_config = _load_config(config)
        
        # Update config with CLI arguments
        user_config.update({
            "device": device,
            "runtime": runtime
        })
        
        click.echo(f"🚀 Starting inference...")
        click.echo(f"   Model: {model_path}")
        click.echo(f"   Input: {input_path}")
        click.echo(f"   Device: {device}, Runtime: {runtime}")
        
        if config:
            click.echo(f"   Config: {config}")
        
        # Initialize inference engine
        start_time = time.time()
        click.echo("⏳ Loading model...")
        
        engine = InferenceEngine(
            model_path=model_path,
            config=user_config,
            device=device,
            runtime=runtime
        )
        
        load_time = time.time() - start_time
        click.echo(f"✅ Model loaded in {load_time:.3f}s")
        
        # Handle different input types
        if input_path.is_file():
            # Single file inference
            if FileUtils.is_image_file(input_path):
                click.echo("🔍 Running single image inference...")
                inference_start = time.time()
                
                result = engine.predict(input_path)
                
                inference_time = time.time() - inference_start
                click.echo(f"✅ Inference completed in {inference_time:.3f}s")
                
                # Add timing information
                result["timing"] = {
                    "model_load_time": load_time,
                    "inference_time": inference_time,
                    "total_time": load_time + inference_time
                }
                
                # Display basic results
                if verbose:
                    click.echo("\n📊 Results:")
                    click.echo(json.dumps(result, indent=2))
                else:
                    # Show summary
                    click.echo(f"\n📊 Inference Summary:")
                    click.echo(f"   Model type: {result.get('model_type', 'unknown')}")
                    click.echo(f"   Outputs: {result.get('num_outputs', 'unknown')}")
                    click.echo(f"   Inference time: {inference_time:.3f}s")
                
                # Save results if output path specified
                if output:
                    _save_results(result, output, format)
                
            else:
                click.echo(f"❌ Unsupported file type: {input_path.suffix}", err=True)
                return
        
        elif input_path.is_dir():
            # Batch processing
            image_files = FileUtils.get_image_files(input_path)
            
            if not image_files:
                click.echo(f"❌ No image files found in: {input_path}", err=True)
                return
            
            click.echo(f"🔍 Running batch inference on {len(image_files)} images...")
            
            batch_results = []
            total_inference_time = 0
            
            with click.progressbar(image_files, label="Processing images") as bar:
                for image_file in bar:
                    try:
                        inference_start = time.time()
                        result = engine.predict(image_file)
                        inference_time = time.time() - inference_start
                        total_inference_time += inference_time
                        
                        result["file_path"] = str(image_file)
                        result["inference_time"] = inference_time
                        batch_results.append(result)
                        
                    except Exception as e:
                        click.echo(f"❌ Failed to process {image_file}: {e}", err=True)
                        continue
            
            # Create batch summary
            summary = {
                "batch_summary": {
                    "total_images": len(image_files),
                    "successful": len(batch_results),
                    "failed": len(image_files) - len(batch_results),
                    "total_inference_time": total_inference_time,
                    "average_inference_time": total_inference_time / len(batch_results) if batch_results else 0,
                    "model_load_time": load_time
                },
                "results": batch_results
            }
            
            click.echo(f"✅ Batch processing completed!")
            click.echo(f"   Processed: {len(batch_results)}/{len(image_files)} images")
            click.echo(f"   Total time: {total_inference_time:.3f}s")
            click.echo(f"   Average: {total_inference_time/len(batch_results):.3f}s per image" if batch_results else "")
            
            # Save batch results
            if output:
                _save_results(summary, output, format)
            elif not verbose:
                # Show summary for each image
                for result in batch_results[:5]:  # Show first 5
                    file_name = Path(result["file_path"]).name
                    click.echo(f"   {file_name}: {result.get('inference_time', 0):.3f}s")
                if len(batch_results) > 5:
                    click.echo(f"   ... and {len(batch_results) - 5} more")
        
        else:
            click.echo(f"❌ Invalid input path: {input_path}", err=True)
            return
    
    except Exception as e:
        click.echo(f"❌ Inference failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return



@click.option("--tag", default="inferx:latest", help="Docker image tag")
@click.option("--template", type=click.Choice(["yolo", "anomalib", "classification"]),
              default="yolo", help="Project template to use")
@click.option("--global", "global_config", is_flag=True, help="Initialize global user config")
@click.pass_context
def init(ctx: click.Context, template: str, global_config: bool) -> None:
    """Initialize new InferX project with template or user config"""
    if global_config:
        # Initialize global user configuration
        # TODO: Implement user config initialization with settings.py
        click.echo("🚧 Global config initialization with settings.py coming soon!")
    else:
        click.echo(f"Initializing project with {template} template")
        # TODO: Implement project initialization
        click.echo("🚧 Project initialization implementation coming soon!")


@cli.command()
@click.option("--model-type", required=True, type=click.Choice(["yolo", "yolo_openvino"]), help="Model type for template")
@click.option("--name", required=True, help="Project name")
@click.option("--device", type=click.Choice(["cpu", "gpu", "auto"]), default="auto", help="Device to run inference on")
@click.option("--runtime", type=click.Choice(["onnx", "openvino", "auto"]), default="auto", help="Runtime engine to use")
@click.option("--with-api", "with_api", is_flag=True, help="Add FastAPI server to template")
@click.option("--with-docker", "with_docker", is_flag=True, help="Add Docker container to template")
@click.option("--config-path", type=click.Path(exists=True), help="Custom configuration file path")
@click.option("--model-path", type=click.Path(exists=True), help="Path to model file to copy into template")
def template(model_type: str, name: str, device: str, runtime: str, with_api: bool, with_docker: bool, config_path: Optional[str], model_path: Optional[str]) -> None:
    """Generate inference project template with optional API and Docker layers"""
    try:
        # Lazy import to avoid dependency issues
        from .generators.template import TemplateGenerator
        
        generator = TemplateGenerator()
        
        # Generate base template
        project_path = generator.create_template(
            model_type, 
            name, 
            with_api=with_api, 
            with_docker=with_docker
        )
        
        # Copy model file if provided
        if model_path:
            generator.copy_model_to_template(model_path, project_path, model_type)
        
        click.echo(f"✅ Generated {model_type} project: {name}")
        if with_api:
            click.echo("   🌐 API server added")
        if with_docker:
            click.echo("   🐳 Docker container added")
        if model_path:
            click.echo(f"   📦 Model copied from: {model_path}")
            
    except ImportError as e:
        click.echo(f"❌ Failed to import template generator: {e}", err=True)
        click.echo("Make sure all dependencies are installed", err=True)
    except Exception as e:
        click.echo(f"❌ Template generation failed: {e}", err=True)
        if "--verbose" in str(e) or "-v" in str(e):
            import traceback
            traceback.print_exc()


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--validate", is_flag=True, help="Validate configuration")
@click.option("--init", is_flag=True, help="Initialize user configuration")
@click.option("--template", type=click.Path(), help="Create config template at specified path")
@click.pass_context  
def config_cmd(ctx: click.Context, show: bool, validate: bool, init: bool, template: Optional[str]) -> None:
    """Configuration management commands"""
    from .settings import get_inferx_settings
    
    if init:
        click.echo("🚧 User config initialization with settings.py coming soon!")
        return
    
    if template:
        click.echo("🚧 Config template creation with settings.py coming soon!")
        return
    
    # Load current config with Pydantic validation
    config = get_inferx_settings()
    
    if validate:
        click.echo("✅ Configuration loaded and validated with Pydantic")
        click.echo(f"   YOLO input size: {config.yolo_input_size}")
        click.echo(f"   Log level: {config.log_level}")
    
    if show:
        import json
        click.echo("📋 Current configuration:")
        # Show key settings
        settings_dict = {
            "yolo_settings": {
                "input_size": config.yolo_input_size,
                "confidence_threshold": config.yolo_confidence_threshold,
                "nms_threshold": config.yolo_nms_threshold
            },
            "device_mapping": config.device_mapping,
            "log_level": config.log_level.value
        }
        click.echo(json.dumps(settings_dict, indent=2))


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--model-path", type=click.Path(exists=True), help="Model file path")
@click.option("--model-type", type=click.Choice(["yolo", "yolo_openvino", "onnx_generic"]), help="Model type")
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8080, help="Port to bind")
def serve(config: Optional[str], model_path: Optional[str], model_type: Optional[str], host: str, port: int) -> None:
    """Start FastAPI server for inference"""
    try:
        # Import here to avoid dependency issues
        from inferx.api.server import start_server
        start_server(config, model_path, model_type, host, port)
    except ImportError as e:
        click.echo(f"❌ Failed to import server module: {e}", err=True)
        click.echo("Make sure all dependencies are installed", err=True)
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}", err=True)


def main() -> None:
    """Main entry point for CLI"""
    cli()