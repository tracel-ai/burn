# Importing Models

Burn supports importing models from other frameworks and file formats, enabling you to use
pre-trained weights in your Burn applications.

## Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| [**ONNX**](./onnx-model.md) | Open Neural Network Exchange format | Direct import of complete model architectures and weights from any framework that supports ONNX export |
| [**Model Weights**](./model-weights.md) | PyTorch (.pt, .pth) and SafeTensors (.safetensors) | Loading tensor weights from PyTorch or Hugging Face models into a matching Burn architecture |
