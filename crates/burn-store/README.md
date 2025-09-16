# Burn Store

> Advanced model storage and serialization for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-store.svg)](https://crates.io/crates/burn-store)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/Tracel-AI/burn/blob/main/LICENSE)

A comprehensive storage library for Burn that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

## Features

### Core Capabilities

- **SafeTensors Format** - Industry-standard format for secure and efficient tensor serialization
- **Zero-Copy Loading** - Memory-mapped files and lazy tensor materialization for optimal
  performance
- **Cross-Framework Support** - Seamless PyTorch ↔ Burn model conversion with automatic adaptations
- **Flexible Filtering** - Load/save specific model subsets with regex, exact paths, or custom
  predicates
- **Tensor Remapping** - Rename tensors during load/save for framework compatibility
- **No-std Support** - Core functionality available in embedded and WASM environments

### Advanced Features

- **Framework Adapters** - Automatic weight transposition and parameter renaming for PyTorch
  compatibility
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
use burn_store::{PyTorchToBurnAdapter, BurnToPyTorchAdapter};

// Load PyTorch model into Burn
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
    .with_key_pattern(r"^old_model\.", "new_model.")  // old_model.X -> new_model.X
    .with_key_pattern(r"\.gamma$", ".weight")         // X.gamma -> X.weight
    .with_key_pattern(r"\.beta$", ".bias");          // X.beta -> X.bias

// Complex remapping with KeyRemapper
use burn_store::KeyRemapper;

let remapper = KeyRemapper::new()
    .add_pattern(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")?  // h.0 -> layer0
    .add_pattern(r"^(.*?)\.attn\.", "$1.attention.")?;                  // attn -> attention

let mut store = SafetensorsStore::from_file("model.safetensors")
    .remap(remapper);
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
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter};
use burn_store::safetensors::SafetensorsStore;

// Load and convert a PyTorch transformer model
let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
    // Automatic PyTorch → Burn conversions
    .with_from_adapter(PyTorchToBurnAdapter)
    // Only load transformer layers
    .with_regex(r"^transformer\..*")
    // Rename layer structure
    .with_key_pattern(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")
    // Handle missing tensors gracefully
    .allow_partial(true)
    // Add conversion metadata
    .metadata("source", "pytorch")
    .metadata("converted_by", "burn-store");

let mut model = TransformerModel::new(&device);
let result = model.apply_from(&mut store)?;

println!("Successfully migrated {} tensors", result.applied.len());
```

## Benchmarks

The crate includes comprehensive benchmarks comparing the new `SafetensorsStore` with the legacy
`SafetensorsFileRecorder`. Benchmarks support multiple backends and include memory allocation
tracking.

### Running Benchmarks

```bash
# Run with default backend (NdArray CPU)
cargo bench --bench safetensor_loading

# Run with specific backend
cargo bench --bench safetensor_loading --features metal    # Apple GPU
cargo bench --bench safetensor_loading --features candle   # Candle backend
cargo bench --bench safetensor_loading --features wgpu     # WGPU
cargo bench --bench safetensor_loading --features cuda     # NVIDIA GPU
cargo bench --bench safetensor_loading --features tch      # LibTorch

# Run with multiple backends
cargo bench --bench safetensor_loading --features "metal candle"
```

### Benchmark Results

The benchmarks test three model sizes (small, medium, large) and show:

- **Execution time** and **throughput** (MB/s)
- **Memory allocation** statistics (max allocations, total allocations/deallocations)
- Comparison between old `SafetensorsFileRecorder` and new `SafetensorsStore`

Typical improvements with the new store:

- **1.75-2.1x faster** loading on CPU backends
- **~60% less memory usage** due to optimized allocation patterns
- Better performance scaling with larger models

## API Overview

### Builder Methods

The `SafetensorsStore` provides a fluent API for configuration:

#### Filtering

- `with_regex(pattern)` - Filter by regex pattern
- `with_full_path(path)` - Include specific tensor
- `with_full_paths(paths)` - Include multiple specific tensors
- `with_predicate(fn)` - Custom filter logic
- `match_all()` - Include all tensors (no filtering)

#### Remapping

- `with_key_pattern(from, to)` - Regex-based renaming
- `remap(KeyRemapper)` - Complex remapping rules

#### Adapters

- `with_from_adapter(adapter)` - Loading transformations
- `with_to_adapter(adapter)` - Saving transformations

#### Configuration

- `metadata(key, value)` - Add custom metadata
- `allow_partial(bool)` - Continue on missing tensors
- `validate(bool)` - Toggle validation
