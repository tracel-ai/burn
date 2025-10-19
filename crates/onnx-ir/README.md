# ONNX-IR

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation (IR) that
can be used to generate code for various ML/DL frameworks. It's a core component of the Burn model
import system, providing a clean abstraction layer between ONNX protobuf structures and Burn's
tensor operations.

## Architecture

The ONNX-IR crate is designed with the following components:

### Core Modules

- **IR Core** (`ir.rs`): Defines the core data structures such as `Node`, `NodeType`, `Argument`,
  `TensorType`, `ValueSource`, etc.
- **Pipeline** (`pipeline.rs`): Main entry point and orchestrator that coordinates the 5-phase
  conversion process
- **Protocol Conversion** (`proto_conversion.rs`): Converts ONNX protobuf structures to IR
  equivalents
- **Graph State** (`graph_state.rs`): Manages mutable state during conversion including node
  storage, name mappings, and tensor data
- **Tensor Store** (`tensor_store.rs`): Centralized storage for tensor data with ID-based access
- **Processor** (`processor.rs`): Defines the `NodeProcessor` trait with type inference and constant
  lifting capabilities
- **Registry** (`registry.rs`): Centralized registry mapping node types to their processors - **add
  new node types here**
- **Node Implementations** (`node/`): Contains 100+ operation-specific processor implementations

### Conversion Phases

The conversion pipeline (`phases/`) consists of five sequential phases:

1. **Initialization** (`initialization.rs`): Creates graph state and processes initializers into
   Constant nodes
2. **Node Conversion** (`node_conversion.rs`): Converts ONNX nodes to IR, performs node remapping,
   and coalesces chains
3. **Type Inference** (`type_inference.rs`): Iteratively infers types with preference propagation
4. **Post-processing** (`post_processing.rs`): Eliminates Identity nodes and lifts constants
5. **Finalization** (`finalization.rs`): Removes unused constants and builds the final graph

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

## Adding New Node Types

To add support for a new ONNX operator:

1. **Create a processor**: Implement `NodeProcessor` trait in a new file under `node/` (e.g.,
   `node/my_op.rs`)
2. **Register the processor**: Add it to `registry.rs` in the `with_standard_processors()` method
3. **Implement type inference**: Define how output types are inferred from inputs
4. **Add constant lifting**: Optionally lift constant inputs to static configuration
5. **Extract config**: Optionally extract codegen configuration from ONNX attributes

See existing processors in `node/` for examples.

## Resources

- **ONNX to Burn Conversion Guide**: For detailed implementation guidance on adding new operators,
  see the
  [ONNX to Burn conversion guide](https://github.com/tracel-ai/burn/blob/main/contributor-book/src/guides/onnx-to-burn-conversion-tool.md).

- **Supported ONNX Operators**: For a full list of currently supported ONNX operators, please see
  the
  [Supported ONNX Operators table](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md).

- **Burn Integration**: ONNX-IR serves as the foundation for the ONNX import support in Burn. The
  conversion from ONNX-IR to Burn graphs is implemented in
  [`burn-import/src/onnx/to_burn.rs`](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/src/onnx/to_burn.rs).
