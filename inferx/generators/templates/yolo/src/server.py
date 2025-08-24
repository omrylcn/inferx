"""FastAPI server for InferX YOLO inference (Template version)"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from pathlib import Path
import tempfile
import yaml
import json
import sys
import os

# Add parent directory to path for imports when run as module
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from .inferencer import Inferencer

# Initialize FastAPI app
app = FastAPI(
    title="InferX YOLO Inference API",
    description="API for YOLO object detection using InferX",
    version="1.0.0"
)

# Load configuration
config_path = Path("config.yaml")
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

# Initialize inferencer
model_path = config.get("model", {}).get("path", "models/yolo_model.onnx")
model_type = config.get("model", {}).get("type", "yolo")

try:
    inferencer = Inferencer(model_path, config)
    print(f"✅ Loaded model: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    inferencer = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "InferX YOLO Inference API", 
        "status": "running",
        "model_path": model_path,
        "model_type": model_type
    }

@app.get("/info")
async def get_model_info():
    """Get model information"""
    if inferencer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        info = inferencer.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Run inference on uploaded image"""
    if inferencer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name
        
        # Run inference
        result = inferencer.predict(tmp_path)
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Run inference on batch of images"""
    if inferencer is None:
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
            
            # Run inference
            result = inferencer.predict(tmp_path)
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

def start_server():
    """Start the FastAPI server"""
    uvicorn.run("src.server:app", host="0.0.0.0", port=8080, reload=False)

if __name__ == "__main__":
    # Add parent directory to path to make imports work when run as module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    uvicorn.run(app, host="0.0.0.0", port=8080)