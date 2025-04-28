# ONNX-IR

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation that can
be used to generate code for various ML/DL frameworks. It's part of the Burn project, with key
features including ONNX model parsing, rank inference, and node remapping. The crate supports
converting ONNX models to Burn graphs and includes utilities for handling constants and graph
transformations.

For a full list of currently supported operators, please check
[here](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md)

## ONNX Compatibility

This library requires ONNX models to use **opset version 16 or higher**. If your model uses an older
opset version, you'll need to upgrade it using the ONNX version converter.

### Upgrading ONNX Models

You can upgrade your ONNX models using the following Python script:

```python
import onnx
from onnx import version_converter, shape_inference

# Load your ONNX model
model = onnx.load('path/to/your/model.onnx')

# Convert the model to opset version 16
upgraded_model = version_converter.convert_version(model, 16)

# Apply shape inference to the upgraded model
inferred_model = shape_inference.infer_shapes(upgraded_model)

# Save the converted model
onnx.save(inferred_model, 'upgraded_model.onnx')
```

For a full list of currently supported operators, please check
[here](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md)

To see how to use this for generating burn graphs, see
[here](crates/burn-import/src/onnx/to_burn.rs).
