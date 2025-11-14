# ONNX-IR

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation (IR) that
can be used to generate code for various ML/DL frameworks. It's a core component of the Burn model
import system, providing a clean abstraction layer between ONNX protobuf structures and Burn's
tensor operations.

## Architecture

ONNX-IR converts ONNX protobuf models into a clean intermediate representation through a 5-phase
pipeline:

1. **Initialization**: Process initializers and create graph state
2. **Node Conversion**: Convert ONNX nodes to IR with node remapping
3. **Type Inference**: Infer output types with preference propagation
4. **Post-processing**: Optimize graph (eliminate Identity nodes, lift constants)
5. **Finalization**: Remove unused nodes and build final `OnnxGraph`

The resulting IR provides:

- **Enum-based node representation**: Each node is a variant of the `Node` enum with operation-specific configuration
- **Typed inputs/outputs**: All node arguments are validated with type information
- **Pre-extracted configuration**: Attributes are parsed into strongly-typed config structs
- **Static tensor data**: Constant values are available for constant folding
- **Support for 100+ ONNX operators**: Including control flow (`If`, `Loop`, `Scan`)

### Node Representation

Nodes are represented using an enum where each variant corresponds to an ONNX operation:

```rust
pub enum Node {
    // Simple operations (no config)
    Add { name: String, inputs: Vec<Argument>, outputs: Vec<Argument> },

    // Operations with configuration
    Softmax {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SoftmaxConfig,
    },

    Conv2d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: Conv2dConfig,
    },

    // ... 100+ more variants
}
```

This enum-based design provides type safety and makes it easy to pattern match on specific operations.

For detailed module documentation, see the inline docs in each module.

## Public API

ONNX-IR exposes a clean public API with three main components:

- **`ir`** module - Core IR types (`OnnxGraph`, `Node`, `Argument`, `TensorType`, `DType`, etc.)
- **`node`** module - Node configurations for all supported operations (e.g., `SoftmaxConfig`, `Conv2dConfig`)
- **`parse_onnx`** - Main parsing function and error types

## Usage

ONNX-IR is typically used through the `burn-import` crate, but can also be used standalone:

```rust
use onnx_ir::{parse_onnx, OnnxGraph, Node};
use std::path::Path;

// Parse an ONNX model into the IR
let graph: OnnxGraph = parse_onnx(Path::new("path/to/model.onnx"));

// Work with the IR - nodes are represented as an enum
for node in &graph.nodes {
    println!("Node: {}", node.name());

    // Pattern match on node type to access operation-specific configuration
    match node {
        Node::Softmax { config, inputs, outputs, .. } => {
            println!("  Softmax on axis {}", config.axis);
            println!("  Inputs: {:?}", inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        }
        Node::Conv2d { config, inputs, outputs, .. } => {
            println!("  Conv2d with {} input channels", config.channels[0]);
            println!("  Kernel size: {:?}", config.kernel_size);
        }
        Node::Add { inputs, outputs, .. } => {
            println!("  Add operation");
        }
        _ => {
            println!("  Other operation");
        }
    }
}

// Access node configurations
use onnx_ir::node::{SoftmaxConfig, Conv2dConfig};

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
