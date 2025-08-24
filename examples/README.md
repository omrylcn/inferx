# InferX Getting Started Examples 🚀

**Quick examples to get you started with InferX in 5 minutes!**

These examples demonstrate the **3 main ways** to use InferX:
1. **📦 Library** - Import and use in Python  
2. **⚡ CLI** - Command line inference
3. **🏗️ Templates** - Generate full projects

## 📦 1. Library Usage Examples

### `01_basic_inference.py` - **START HERE!**
```bash
uv run python examples/01_basic_inference.py
```
**What you'll learn:**
- Load a YOLO model and run inference  
- Basic ONNX vs OpenVINO usage
- Process single images
- **5 minutes to working inference!**

### `02_batch_processing.py` 
```bash
uv run python examples/02_batch_processing.py
```
**What you'll learn:**
- Process multiple images at once
- Performance comparison
- Output results to files

## ⚡ 2. CLI Usage Examples

### `03_cli_examples.py` - **Interactive CLI Demo**
```bash
uv run python examples/03_cli_examples.py
```
**What you'll learn:**
- Run models from command line
- Different CLI options and flags
- Batch processing with CLI
- **See CLI in action!**

## 🏗️ 3. Template Generation Examples

### `04_template_walkthrough.py` - **Project Generation Demo**
```bash
uv run python examples/04_template_walkthrough.py
```
**What you'll learn:**
- Generate all 4 template types
- Set up projects with UV
- Run API servers
- **Go from zero to deployed API in minutes!**

## 🎯 Quick Start Workflow

**New to InferX? Follow this path:**

1. **📦 Start Here**: `01_basic_inference.py` - Learn the basics (5 min)
2. **⚡ Try CLI**: `03_cli_examples.py` - See command line usage (3 min)  
3. **🏗️ Generate Project**: `04_template_walkthrough.py` - Create full projects (5 min)
4. **🚀 Deploy**: You now have working inference + API server!

**Total time: ~15 minutes from zero to deployed API! 🚀**

## 📋 What's in Each Example

| Example | Focus | Time | What You'll Build |
|---------|-------|------|-------------------|
| `01_basic_inference.py` | 📦 Library basics | 5 min | Working YOLO inference |
| `02_batch_processing.py` | 📦 Batch processing | 3 min | Multi-image processing |  
| `03_cli_examples.py` | ⚡ CLI usage | 3 min | Command-line skills |
| `04_template_walkthrough.py` | 🏗️ Project generation | 5 min | Full API project |

## 📚 Next Steps After Examples

1. **📖 Read**: `USAGE.md` for complete feature guide
2. **🛠️ Build**: Create your own project with templates
3. **🚀 Deploy**: Use generated API servers in production
4. **📞 Support**: Check `README.md` for help

## 🎯 Legacy Files (Skip for Getting Started)

Legacy examples have been moved to `legacy/` directory:
- `legacy/basic_usage.py` - Comprehensive legacy example
- `legacy/settings_examples.py` - Advanced configuration  
- `legacy/yolo_detection_demo.py` - Detailed YOLO demo
- `legacy/example_config.yaml` - Advanced configuration file

**For getting started, use the numbered examples (01-04) in this directory! 🚀**

⚠️ **Note**: If you see legacy files in the main directory (not in `legacy/`), you can safely remove:
- `basic_usage.py`
- `settings_examples.py` 
- `yolo_detection_demo.py`
- `example_config.yaml`

These are outdated and have been replaced by the modern numbered examples.