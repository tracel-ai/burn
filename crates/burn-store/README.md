# Burn Store

> Advanced model storage and serialization for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-store.svg)](https://crates.io/crates/burn-store)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/Tracel-AI/burn/blob/main/LICENSE)

A comprehensive storage library for Burn that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

## Features

### Core Capabilities

- **SafeTensors Format** - Industry-standard format for secure and efficient tensor serialization
- **PyTorch Support** - Direct loading of PyTorch .pth/.pt files with automatic weight transformation
- **Zero-Copy Loading** - Memory-mapped files and lazy tensor materialization for optimal performance
- **Cross-Framework Support** - Seamless PyTorch ↔ Burn model conversion with automatic adaptations
- **Flexible Filtering** - Load/save specific model subsets with regex, exact paths, or custom predicates
- **Tensor Remapping** - Rename tensors during load/save for framework compatibility
- **No-std Support** - Core functionality available in embedded and WASM environments

### Advanced Features

- **Framework Adapters** - Automatic weight transposition and parameter renaming for PyTorch compatibility
- **Lazy Transformations** - Chain tensor transformations without materializing intermediate data
- **Partial Loading** - Continue loading even when some tensors are missing
- **Custom Metadata** - Attach version info, training details, or other metadata to saved models
- **Efficient Inspection** - Query tensor shapes and dtypes without loading data

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-store = "0.19"

# Optional features
burn-store = { version = "0.19", features = ["std", "safetensors"] }
```

## Quick Start

### Basic Save and Load

```rust
use burn_store::{ModuleSnapshot, SafetensorsStore};

// Save a model
let mut store = SafetensorsStore::from_file("model.safetensors");
model.collect_to(&mut store)?;

// Load a model
let mut store = SafetensorsStore::from_file("model.safetensors");
model.apply_from(&mut store)?;
```

### Filtering Tensors

```rust
// Save only encoder layers
let mut store = SafetensorsStore::from_file("encoder.safetensors")
    .with_regex(r"^encoder\..*")
    .metadata("subset", "encoder_only");

model.collect_to(&mut store)?;

// Load with multiple filter patterns (OR logic)
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")      // Include encoder tensors
    .with_regex(r".*\.bias$")          // OR include any bias tensors
    .with_full_path("decoder.scale"); // OR include specific tensor

model.apply_from(&mut store)?;
```

### PyTorch Interoperability

```rust
use burn_store::{PyTorchToBurnAdapter, BurnToPyTorchAdapter, PytorchStore};

// Load PyTorch .pth file directly (PyTorchToBurnAdapter is applied automatically)
let mut store = PytorchStore::from_file("pytorch_model.pth")
    .with_top_level_key("state_dict")         // Access nested state dict
    .allow_partial(true);                     // Skip unknown tensors

burn_model.apply_from(&mut store)?;

// Load PyTorch model from SafeTensors
let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter)  // Auto-transpose linear weights
    .allow_partial(true);                     // Skip unknown PyTorch tensors

burn_model.apply_from(&mut store)?;

// Save Burn model for PyTorch
let mut store = SafetensorsStore::from_file("for_pytorch.safetensors")
    .with_to_adapter(BurnToPyTorchAdapter);   // Convert back to PyTorch format

burn_model.collect_to(&mut store)?;
```

### Tensor Name Remapping

```rust
// Simple pattern-based remapping
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_key_remapping(r"^old_model\.", "new_model.")  // old_model.X -> new_model.X
    .with_key_remapping(r"\.gamma$", ".weight")         // X.gamma -> X.weight
    .with_key_remapping(r"\.beta$", ".bias");           // X.beta -> X.bias

// Complex remapping with KeyRemapper
use burn_store::KeyRemapper;

let remapper = KeyRemapper::new()
    .add_pattern(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")?  // h.0 -> layer0
    .add_pattern(r"^(.*?)\.attn\.", "$1.attention.")?;                  // attn -> attention

let mut store = SafetensorsStore::from_file("model.safetensors")
    .remap(remapper);

// Combining with PyTorch loading
let mut store = PytorchStore::from_file("model.pth")
    .with_key_remapping(r"^model\.", "")           // Remove model. prefix
    .with_key_remapping(r"norm(\d+)", "norm_$1");  // norm1 -> norm_1
```

### Memory Operations

```rust
// Save to memory buffer
let mut store = SafetensorsStore::from_bytes(None)
    .with_regex(r"^encoder\..*");
model.collect_to(&mut store)?;
let bytes = store.get_bytes()?;

// Load from memory buffer
let mut store = SafetensorsStore::from_bytes(Some(bytes))
    .allow_partial(true);
let result = model.apply_from(&mut store)?;

println!("Loaded {} tensors", result.applied.len());
if !result.missing.is_empty() {
    println!("Missing: {:?}", result.missing);
}
```

### Complete Example: Migrating PyTorch Models

```rust
use burn_store::{ModuleSnapshot, PytorchStore};

// Load directly from PyTorch .pth file (automatic PyTorchToBurnAdapter)
let mut store = PytorchStore::from_file("pytorch_transformer.pth")
    // Access the state dict
    .with_top_level_key("state_dict")
    // Only load transformer layers
    .with_regex(r"^transformer\..*")
    // Rename layer structure to match Burn model
    .with_key_remapping(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")
    // Rename attention layers
    .with_key_remapping(r"\.attn\.", ".attention.")
    // Handle missing tensors gracefully
    .allow_partial(true);

let mut model = TransformerModel::new(&device);
let result = model.apply_from(&mut store)?;

println!("Successfully migrated {} tensors", result.applied.len());
if !result.errors.is_empty() {
    println!("Errors: {:?}", result.errors);
}

// Save the migrated model in SafeTensors format
let mut save_store = SafetensorsStore::from_file("migrated_model.safetensors")
    .metadata("source", "pytorch")
    .metadata("converted_by", "burn-store");

model.collect_to(&mut save_store)?;
```

## Advanced Usage

### Custom Filtering with Predicates

```rust
// Custom filter function
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_predicate(|path, _container| {
        // Only load tensors with specific characteristics
        path.contains("weight") && !path.contains("bias")
    });
```

### Working with Containers

```rust
// Filter based on container types (Linear, Conv2d, etc.)
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_predicate(|_path, container| {
        // Only load Linear layer parameters
        container.split('.').last() == Some("Linear")
    });
```

### Handling Load Results

```rust
let result = model.apply_from(&mut store)?;

// Detailed result information
println!("Applied: {} tensors", result.applied.len());
println!("Skipped: {} tensors", result.skipped.len());
println!("Missing: {:?}", result.missing);
println!("Unused: {:?}", result.unused);

if !result.errors.is_empty() {
    for (path, error) in &result.errors {
        eprintln!("Error loading {}: {}", path, error);
    }
}
```

## Benchmarks

```bash
# Generate model files first (one-time setup)
cd crates/burn-store
uv run benches/generate_unified_models.py

# Run unified benchmark with default backend (NdArray CPU)
cargo bench --bench unified_loading

# Run with specific backend
cargo bench --bench unified_loading --features metal    # Apple GPU
cargo bench --bench unified_loading --features wgpu     # WebGPU
cargo bench --bench unified_loading --features cuda     # NVIDIA GPU
cargo bench --bench unified_loading --features candle   # Candle backend
cargo bench --bench unified_loading --features tch      # LibTorch

# Run with multiple backends
cargo bench --bench unified_loading --features "wgpu metal"
```

## API Overview

### Builder Methods

The stores provide a fluent API for configuration:

#### Filtering

- `with_regex(pattern)` - Filter by regex pattern
- `with_full_path(path)` - Include specific tensor
- `with_full_paths(paths)` - Include multiple specific tensors
- `with_predicate(fn)` - Custom filter logic
- `match_all()` - Include all tensors (no filtering)

#### Remapping

- `with_key_remapping(from, to)` - Regex-based tensor renaming
- `remap(KeyRemapper)` - Complex remapping rules

#### Adapters

- `with_from_adapter(adapter)` - Loading transformations
- `with_to_adapter(adapter)` - Saving transformations

#### Configuration

- `metadata(key, value)` - Add custom metadata (SafeTensors only)
- `allow_partial(bool)` - Continue on missing tensors
- `validate(bool)` - Toggle validation
- `with_top_level_key(key)` - Access nested dict in PyTorch files

## License

This project is dual-licensed under MIT and Apache-2.0. See [LICENSE](https://github.com/Tracel-AI/burn/blob/main/LICENSE) for details.
