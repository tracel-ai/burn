# Burn Import

The `burn-import` crate enables seamless integration of pre-trained models from popular machine
learning frameworks into the Burn ecosystem. This functionality allows you to leverage existing
models while benefiting from Burn's performance optimizations and native Rust integration.

## Supported Import Formats

Burn currently supports three primary model import formats, each serving different use cases:

| Format                                                                              | Description                               | Use Case                                                                                               |
| ----------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [**ONNX** (Guide)](https://burn.dev/books/burn//import/onnx-model.html)               | Open Neural Network Exchange format       | Direct import of complete model architectures and weights from any framework that supports ONNX export |
| [**PyTorch** (Guide)](https://burn.dev/books/burn//import/pytorch-model.html)         | PyTorch weights (.pt, .pth)               | Loading weights from PyTorch models into a matching Burn architecture                                  |
| [**Safetensors** (Guide)](https://burn.dev/books/burn//import/safetensors-model.html) | Hugging Face's model serialization format | Loading a model's tensor weights into a matching Burn architecture                                     |

## ONNX Contributor Resources

- [ONNX to Burn conversion guide](https://burn.dev/books/contributor/guides/onnx-to-burn-conversion-tool.html) -
  Instructions for adding support for additional ONNX operators
- [ONNX tests README](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/onnx-tests/README.md) -
  Testing procedures for ONNX operators
- [Supported ONNX Operators table](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md) -
  Complete list of currently supported ONNX operators
