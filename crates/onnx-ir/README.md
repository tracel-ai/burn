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

- **Enum-based node representation**: Each node is a variant of the `Node` enum with
  operation-specific configuration
- **Typed inputs/outputs**: All node arguments are validated with type information
- **Pre-extracted configuration**: Attributes are parsed into strongly-typed config structs
- **Static tensor data**: Constant values are available for constant folding
- **Support for 100+ ONNX operators**: Including control flow (`If`, `Loop`, `Scan`)

### Node Representation

Nodes are represented using an enum where each variant wraps an operation-specific node struct:

```rust
pub enum Node {
    Add(arithmetic::AddNode),
    Softmax(softmax::SoftmaxNode),
    Conv2d(conv2d::Conv2dNode),
    // ... 200+ more variants
}

// Each node struct contains name, inputs, outputs, and optional config
pub struct SoftmaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SoftmaxConfig,
}
```

This design provides type safety, enables trait implementations on specific node types, and uses a
unified macro (`define_node_enum!`) to generate both `NodeType` and `Node` enums from a single
source of truth.

For detailed module documentation, see the inline docs in each module.

## Public API

ONNX-IR exposes a clean public API with three main components:

- **`ir`** module - Core IR types (`OnnxGraph`, `Node`, `Argument`, `TensorType`, `DType`, etc.)
- **`node`** module - Node configurations for all supported operations (e.g., `SoftmaxConfig`,
  `Conv2dConfig`)
- **`OnnxGraphBuilder`** - Builder for parsing ONNX models from files, bytes, or readers
- **`Error`** - Error type for parsing failures

## Usage

ONNX-IR is typically used through the `burn-onnx` crate, but can also be used standalone:

```rust
use onnx_ir::{OnnxGraphBuilder, OnnxGraph, Node};

// Parse an ONNX model from file (uses mmap when available)
let graph: OnnxGraph = OnnxGraphBuilder::new()
    .parse_file("path/to/model.onnx")?;

// Or parse from bytes
let graph = OnnxGraphBuilder::new().parse_bytes(&model_bytes)?;

// Or parse from a reader
let graph = OnnxGraphBuilder::new().parse_reader(file)?;

// Work with the IR - nodes are represented as an enum
for node in &graph.nodes {
    println!("Node: {}", node.name());

    // Pattern match on node type to access operation-specific configuration
    match node {
        Node::Softmax(softmax_node) => {
            println!("  Softmax on axis {}", softmax_node.config.axis);
            println!("  Inputs: {:?}", softmax_node.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        }
        Node::Conv2d(conv_node) => {
            println!("  Conv2d with {} input channels", conv_node.config.channels[0]);
            println!("  Kernel size: {:?}", conv_node.config.kernel_size);
        }
        Node::Add(add_node) => {
            println!("  Add operation with {} inputs", add_node.inputs.len());
        }
        _ => {
            println!("  Other operation");
        }
    }
}

// Access node configurations
use onnx_ir::node::{SoftmaxConfig, Conv2dConfig};

// Convert to another framework's representation
// (This is typically done by burn-onnx or another conversion layer)
```

## Memory-Mapped Loading

By default, ONNX-IR uses memory-mapped file I/O (mmap) when loading models from files. This provides:

- **Reduced memory usage**: Tensor data is read directly from the file on demand
- **Faster startup**: No need to copy the entire file into memory upfront
- **Lazy loading**: Data is only copied when actually accessed

The `mmap` feature is enabled by default. To disable it:

```toml
[dependencies]
onnx-ir = { version = "...", default-features = false }
```

When parsing from bytes or readers, the data is copied into memory (mmap only applies to file paths).

## ONNX Compatibility

This library recommends ONNX models use **opset version 16 or higher** for best compatibility. While
models with older opset versions may work, opset 16+ ensures access to all supported operators and
their latest behavior. If you encounter issues with an older model, consider upgrading it using the
ONNX version converter.

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
  [Supported ONNX Operators table](https://github.com/tracel-ai/burn/blob/main/crates/burn-onnx/SUPPORTED-ONNX-OPS.md).

- **Burn Integration**: ONNX-IR serves as the foundation for the ONNX import support in Burn. The
  conversion from ONNX-IR to Burn graphs is implemented in
  [`burn-onnx/src/burn/`](https://github.com/tracel-ai/burn/blob/main/crates/burn-onnx/src/burn/).
