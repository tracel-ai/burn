# Importing Models

The Burn project provides robust support for importing models from various machine learning frameworks, allowing you to leverage pre-trained models and weights in your Burn applications. These import capabilities are designed with efficiency, security, and compatibility in mind.

## Supported Formats

Burn currently supports three primary model import formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| [**ONNX**](./onnx-model.md) | Open Neural Network Exchange format | Direct import of complete model architectures and weights from any framework that supports ONNX export |
| [**PyTorch**](./pytorch-model.md) | PyTorch weights (.pt, .pth) | Loading weights from PyTorch models into a matching Burn architecture |
| [**Safetensors**](./safetensors-model.md) | Hugging Face's secure tensor format | Secure, efficient, and language-agnostic weight loading without pickle dependencies |
