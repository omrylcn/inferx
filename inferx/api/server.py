"""Standalone FastAPI server for InferX inference"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import yaml
from pathlib import Path
import tempfile
import sys
import os

def load_config(config_path: str = None):
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    # Look for default config files
    for config_file in ["config.yaml", "inferx_config.yaml"]:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
    
    # Default configuration
    return {
        "model": {
            "path": "models/yolo_model.onnx",
            "type": "yolo"
        }
    }

def create_app(config_path: str = None, model_path: str = None, model_type: str = None):
    """Create FastAPI app with InferX inference"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Override with command line arguments if provided
    if model_path:
        config.setdefault("model", {})["path"] = model_path
    if model_type:
        config.setdefault("model", {})["type"] = model_type
    
    # Get model configuration
    model_config = config.get("model", {})
    model_path = model_config.get("path", "models/yolo_model.onnx")
    model_type = model_config.get("type", None)  # Let runtime auto-detect if None
    
    # Initialize inference engine using runtime.py
    try:
        from inferx.runtime import InferenceEngine
        engine = InferenceEngine(
            model_path=model_path,
            model_type=model_type, 
            config=config,
            device="auto",
            runtime="auto"
        )
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        engine = None
    
    # Initialize FastAPI app
    app = FastAPI(
        title="InferX Inference API",
        description="API for model inference using InferX",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {
            "message": "InferX Inference API", 
            "status": "running",
            "model_path": model_path,
            "model_type": model_type
        }
    
    @app.get("/info")
    async def get_model_info():
        """Get model information"""
        if engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Use runtime.py's get_model_info method
            info = engine.get_model_info()
            return info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
    
    @app.post("/predict")
    async def predict_image(file: UploadFile = File(...)):
        """Run inference on uploaded image"""
        if engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                tmp_file.write(await file.read())
                tmp_path = tmp_file.name
            
            # Run inference using runtime.py
            result = engine.predict(tmp_path)
            
            # Clean up temporary file
            Path(tmp_path).unlink()
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    @app.post("/predict-batch")
    async def predict_batch(files: list[UploadFile] = File(...)):
        """Run inference on batch of images"""
        if engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        results = []
        failed_files = []
        
        for file in files:
            try:
                # Validate file type
                if not file.content_type.startswith("image/"):
                    failed_files.append({"filename": file.filename, "error": "Not an image file"})
                    continue
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                    tmp_file.write(await file.read())
                    tmp_path = tmp_file.name
                
                # Run inference using runtime.py
                result = engine.predict(tmp_path)
                result["filename"] = file.filename
                results.append(result)
                
                # Clean up temporary file
                Path(tmp_path).unlink()
                
            except Exception as e:
                failed_files.append({"filename": file.filename, "error": str(e)})
        
        return {
            "results": results,
            "failed": failed_files,
            "total_processed": len(results),
            "total_failed": len(failed_files)
        }
    
    return app

def start_server(config_path: str = None, model_path: str = None, model_type: str = None, host: str = "0.0.0.0", port: int = 8080):
    """Start the FastAPI server"""
    app = create_app(config_path, model_path, model_type)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="InferX Inference Server")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--model-path", help="Model file path")
    parser.add_argument("--model-type", help="Model type")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    
    args = parser.parse_args()
    start_server(args.config, args.model_path, args.model_type, args.host, args.port)