# ONNX-IR

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation (IR) that
can be used to generate code for various ML/DL frameworks. It's a core component of the Burn model
import system, providing a clean abstraction layer between ONNX protobuf structures and Burn's
tensor operations.

## Architecture

The ONNX-IR crate is designed with the following components:

- **IR Core** (`ir.rs`): Defines the core data structures such as `Node`, `NodeType`, `Argument`,
  etc.
- **Protocol Conversion** (`proto_conversion.rs`): Converts ONNX protobuf structures to IR
- **ONNX Parsing** (`from_onnx.rs`): Handles the parsing of ONNX models into the IR
- **Rank Inference** (`rank_inference.rs`): Computes output tensor ranks for each operation
- **Node Implementations** (`node/`): Contains operation-specific configurations and rank inference
  functions
- **Node Remapping** (`node_remap.rs`): Maps generic ONNX operations to dimension-specific
  alternatives

## Usage

ONNX-IR is typically used through the `burn-import` crate, but can also be used standalone:

```rust
use onnx_ir::{parse_onnx, OnnxGraph};
use std::path::Path;

// Parse an ONNX model into the IR
let graph: OnnxGraph = parse_onnx(Path::new("path/to/model.onnx"));

// Work with the IR
for node in &graph.nodes {
    println!("Node: {}, Type: {:?}", node.name, node.node_type);

    // Access inputs and outputs
    for input in &node.inputs {
        println!("  Input: {}", input.name);
    }

    for output in &node.outputs {
        println!("  Output: {}", output.name);
    }
}

// Convert to another framework's representation
// (This is typically done by burn-import or another conversion layer)
```

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

## Adding New Operators

To add support for a new ONNX operator:

1. Add a new NodeType variant in `ir.rs`
2. Create a new module in `node/<operation_name>.rs` with:
   - A configuration function that extracts parameters from ONNX nodes
   - A rank inference function that updates output tensor ranks
3. Register the rank inference function in `rank_inference.rs`
4. If the operation works with constants, add it to `LIFT_CONSTANTS_FOR_NODE_TYPES` in
   `from_onnx.rs`

For detailed implementation guidance, see the
[ONNX to Burn conversion guide](https://github.com/tracel-ai/burn/blob/main/contributor-book/src/guides/onnx-to-burn-conversion-tool.md).

## Supported Operators

For a full list of currently supported ONNX operators, please see the
[Supported ONNX Operators table](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).

## Integration with Burn

ONNX-IR serves as the foundation for the ONNX import support in Burn. The conversion from ONNX-IR to
Burn graphs is implemented in
[`burn-import/src/onnx/to_burn.rs`](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/src/onnx/to_burn.rs).
