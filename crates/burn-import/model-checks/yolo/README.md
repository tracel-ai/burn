# YOLO Model Checks

This crate provides a unified interface for testing multiple YOLO model variants with Burn.

## Supported Models

- `yolov5s` - YOLOv5 small variant
- `yolov8n` - YOLOv8 nano variant
- `yolov8s` - YOLOv8 small variant
- `yolo11x` - YOLO11 extra-large variant
- `yolov10n` - YOLOv10 nano variant (Note: Currently fails due to TopK operator issue)

## Usage

### 1. Download and prepare a model

```bash
# Using Python directly
python get_model.py --model yolov8n

# Or using uv
uv run get_model.py --model yolov8n

# List available models
uv run get_model.py --list
```

### 2. Run the model test

After building, you can run the test. The model is already compiled in:

```bash
YOLO_MODEL=yolov8s cargo run --release
```

## Directory Structure

```
yolo/
├── artifacts/           # Downloaded ONNX models and test data
│   ├── yolov8n_opset16.onnx
│   ├── yolov8n_test_data.pt
│   └── ...
├── src/
│   └── main.rs         # Test runner
├── build.rs            # Build script that generates model code
├── get_model.py        # Model download and preparation script
└── Cargo.toml
```

## Notes

- All YOLO models (except v10) output shape `[1, 84, 8400]` for standard object detection
- YOLOv10n has a different architecture with output shape `[1, 300, 6]` and uses TopK operator
- The crate requires explicit model selection at build time (no default model)
