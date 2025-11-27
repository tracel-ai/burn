# RF-DETR Model Check

This crate tests burn-import's ability to handle the RF-DETR (Roboflow DETR) object detection model.

## About RF-DETR

RF-DETR is a real-time object detection model based on the DETR (Detection Transformer)
architecture, developed by Roboflow. It combines transformer-based detection with optimizations for
speed and accuracy.

Key features:

- Transformer-based architecture with multi-head attention
- Deformable attention mechanisms
- Object queries for detection
- End-to-end trainable without anchor boxes or NMS

## Related Issue

This model check was created to track and test the fix for:

- [Issue #4052](https://github.com/tracel-ai/burn/issues/4052): RF-DETR ONNX import fails with "axis
  2 is out of bounds for rank 1"

## Usage

### 1. Download and prepare the model

Requires Python 3.11:

```bash
# Using uv (recommended)
uv run --python 3.11 get_model.py
```

### 2. Build and run the model test

```bash
cargo build
cargo run
```

## Directory Structure

```
rf-detr/
├── artifacts/                        # Downloaded ONNX model and test data
│   ├── rf_detr_small.onnx           # ONNX model (119 MB)
│   ├── rf_detr_small_test_data.pt   # Test input/output tensors (3.1 MB)
│   └── node_info.json               # ONNX node analysis
├── src/
│   └── main.rs                      # Test runner with output comparison
├── build.rs                         # Build script that generates model code
├── get_model.py                     # Model download, export, and test data generation
└── Cargo.toml
```

## Model Details

- **Model**: RF-DETR Small
- **Input**: `[1, 3, 512, 512]` (RGB image)
- **Outputs**:
  - `dets`: `[1, 300, 4]` - 300 bounding boxes (x, y, w, h)
  - `labels`: `[1, 300, 91]` - 300 class scores (91 COCO classes)
- **Architecture**: DETR with deformable attention

## Test Data

The `get_model.py` script generates reference test data by:

1. Creating a reproducible random input tensor (seed 42)
2. Running inference with ONNX Runtime
3. Saving both input and outputs as PyTorch tensors

When the ONNX import issue is fixed, `cargo run` will:

1. Load the test data
2. Run inference with the burn-generated model
3. Compare outputs against ONNX Runtime reference within tolerance (1e-4)

## Notes

- The model uses transformer layers which test burn-import's handling of attention mechanisms
- The model includes complex operations like multi-head attention and deformable convolutions
- Currently fails at build time due to issue #4052 (axis out of bounds during type inference)
