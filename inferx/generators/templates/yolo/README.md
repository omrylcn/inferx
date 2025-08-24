# YOLO ONNX Inference Project

This is an auto-generated YOLO inference project using InferX.

## 🚀 Quick Start

### Without API Server

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Add your YOLO ONNX model:**
   Place your YOLO model file in the `models/` directory and rename it to `yolo_model.onnx`.

3. **Run inference:**
   ```bash
   uv run python src/inferencer.py path/to/your/image.jpg
   ```

### With API Server

1. **Install dependencies with API support:**
   ```bash
   uv sync --extra api
   ```

2. **Add your YOLO ONNX model:**
   Place your YOLO model file in the `models/` directory and rename it to `yolo_model.onnx`.

3. **Start the API server:**
   ```bash
   uv run --extra api python -m src.server
   ```

4. **Access the API:**
   Open your browser and go to `http://localhost:8080/docs` for API documentation.

## 📁 Project Structure

```
.
├── models/                 # Model files (place your YOLO model here)
├── src/                    # Source code
│   ├── __init__.py
│   ├── base.py             # Base inferencer class
│   ├── exceptions.py       # Custom exceptions
│   ├── inferencer.py       # YOLO ONNX inferencer
│   ├── server.py           # FastAPI server (if API is enabled)
│   ├── utils.py            # Utility functions
│   └── yolo_base.py        # YOLO base class
├── config.yaml             # Configuration file
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # This file
```

## ⚙️ Configuration

The `config.yaml` file contains the configuration for the YOLO model:

```yaml
model:
  path: "models/yolo_model.onnx"
  type: "yolo"

inference:
  device: "auto"
  runtime: "auto"
  confidence_threshold: 0.25
  nms_threshold: 0.45
  input_size: 640

preprocessing:
  target_size: [640, 640]
  normalize: true
  color_format: "RGB"
  maintain_aspect_ratio: true
```

## 🧪 Testing

### Without API Server

To test the YOLO inference:

```bash
uv run python src/inferencer.py path/to/test/image.jpg
```

### With API Server

To test the API server:

1. Start the server:
   ```bash
   uv run --extra api python -m src.server
   ```

2. Send a POST request to `/predict` with an image file:
   ```bash
   curl -X POST "http://localhost:8080/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/test/image.jpg"
   ```

## 🐳 Docker Support

To build and run with Docker:

```bash
docker build -t yolo-inference .
docker run -p 8080:8080 yolo-inference
```

## 📚 Documentation

For more information about InferX, see:
- [InferX GitHub Repository](https://github.com/omrylcn/inferx)
- [Usage Guide](https://github.com/omrylcn/inferx/blob/main/USAGE.md)