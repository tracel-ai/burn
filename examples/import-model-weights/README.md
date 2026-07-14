# Import Model Weights

This crate provides examples for importing model weights from different formats to Burn.

## Examples

### PyTorch

Imports weights from a PyTorch `.pt` file using `burn-store`.

```bash
cargo run --bin pytorch -- <image_index>
```

Example:
```bash
cargo run --bin pytorch -- 15

Loading PyTorch model weights from file: weights/mnist.pt
Image index: 15
Success!
Predicted: 5
Actual: 5
See the image online, click the link below:
https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row=15
```

### Safetensors

Imports weights from a Safetensors file using `burn-store`.

```bash
cargo run --bin safetensors -- <image_index>
```

Example:
```bash
cargo run --bin safetensors -- 42

Loading Safetensors model weights from file: weights/mnist.safetensors
Image index: 42
Success!
Predicted: 4
Actual: 4
See the image online, click the link below:
https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row=42
```

### Convert

Converts between different weight formats (PyTorch or Safetensors) to Burn's native Burnpack format.

```bash
cargo run --bin convert -- <format> <output_directory>
```

Where:
- `<format>`: Either `pytorch` or `safetensors`
- `<output_directory>`: Path to save the converted model file

Example with PyTorch:
```bash
cargo run --bin convert -- pytorch /tmp/burn-convert

Loading PyTorch weights from 'weights/mnist.pt'...
Saving model to '/tmp/burn-convert/mnist.bpk'...
Model successfully saved to '/tmp/burn-convert/mnist.bpk'.
```

Example with Safetensors:
```bash
cargo run --bin convert -- safetensors /tmp/burn-convert

Loading Safetensors weights from 'weights/mnist.safetensors'...
Saving model to '/tmp/burn-convert/mnist.bpk'...
Model successfully saved to '/tmp/burn-convert/mnist.bpk'.
```

### Burnpack

Demonstrates loading and using a model from Burn's native Burnpack format.

```bash
cargo run --bin burnpack -- <image_index> <model_path>
```

Where:
- `<image_index>`: Index of the MNIST test image to classify
- `<model_path>`: Path to the model file (without extension)

Example:
```bash
cargo run --bin burnpack -- 35 /tmp/burn-convert/mnist

Loading model weights from file: /tmp/burn-convert/mnist.bpk
Image index: 35
Success!
Predicted: 2
Actual: 2
See the image online, click the link below:
https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row=35
```

## Workflow

A typical workflow using these examples:

1. Start with pre-trained weights in either PyTorch or Safetensors format
2. Use the `convert` example to convert to Burn's native Burnpack format
3. Load and use the converted model with the `burnpack` example
