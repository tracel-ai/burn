# Importing Models

Burn supports importing models from other frameworks and file formats, enabling you to use pre-trained weights in your Burn applications.

## Supported Formats

Burn currently supports three primary model import formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| [**ONNX**](./onnx-model.md) | Open Neural Network Exchange format | Direct import of complete model architectures and weights from any framework that supports ONNX export |
| [**PyTorch**](./pytorch-model.md) | PyTorch weights (.pt, .pth) | Loading weights from PyTorch models into a matching Burn architecture |
| [**Safetensors**](./safetensors-model.md) | Hugging Face's secure tensor format | Secure, efficient, and language-agnostic weight loading without pickle dependencies |
