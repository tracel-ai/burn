# Burn Import

The `burn-import` crate provides tools for importing models from other machine learning frameworks
into the Burn ecosystem. It allows users to leverage pre-trained models from popular frameworks
while benefiting from Burn's performance and Rust integration.

## Supported Formats

### ONNX

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) is an open standard for machine learning
interoperability. Burn supports importing ONNX models with opset version 16 or higher, converting
them to native Burn code and model weights.

- **Convert models from**: PyTorch, TensorFlow, Keras, scikit-learn, and other ONNX-compatible
  frameworks
- **Full code generation**: Generates Rust source code that matches the original model's
  architecture
- **Complete state handling**: Extracts and converts model weights to Burn's format

See the [ONNX import documentation](https://burn.dev/burn-book/import/onnx-model.html) for usage
details.

### PyTorch

Burn supports direct import of PyTorch model weights (.pt/.pth files) into Burn model architectures:

- **Direct weight loading**: Map PyTorch layer weights to equivalent Burn layers
- **Efficient conversion**: No need for ONNX as an intermediate format
- **Compatible with**: Common PyTorch architectures and custom models

See the [PyTorch import documentation](https://burn.dev/burn-book/import/pytorch-model.html) for
usage details.

## Extending Support

### Adding New ONNX Operators

The crate is designed to be extensible. To add support for new ONNX operators:

1. Implement the operator in the `onnx-ir` crate
2. Add the operator conversion logic in `src/onnx/to_burn.rs`
3. Register the operator in the conversion pipeline

See our
[ONNX to Burn conversion guide](https://github.com/tracel-ai/burn/blob/main/contributor-book/src/guides/onnx-to-burn-conversion-tool.md)
for detailed instructions.

### Adding New Import Formats

To add support for a new model format:

1. Create a new module under `src/` for the format
2. Implement the parsing and conversion logic
3. Add CLI support for the new format

## Testing

The `onnx-tests` subcrate contains comprehensive tests for the ONNX import functionality:

- **Unit tests**: Verify specific operator conversions
- **End-to-end tests**: Ensure complete models are correctly imported
- **Comparison tests**: Validate that imported models produce the same outputs as original models

See the
[ONNX tests README](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/onnx-tests/README.md)
for details on testing.

## Supported ONNX Operators

For a complete list of supported ONNX operators, see the
[Supported ONNX Operators table](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).
