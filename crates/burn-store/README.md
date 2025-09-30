# Burn Store

> Advanced model storage and serialization for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-store.svg)](https://crates.io/crates/burn-store)

A comprehensive storage library for Burn that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

## Features

### Core Capabilities

- **SafeTensors Format** - Industry-standard format for secure and efficient tensor serialization
- **PyTorch Support** - Direct loading of PyTorch .pth/.pt files with automatic weight
  transformation
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

SafetensorsStore supports no-std environments when using byte operations

### Model Surgery and Partial Operations

Burn Store enables sophisticated model surgery operations for selectively loading, saving, and
transferring parts of models.

#### Direct Model-to-Model Transfer

```rust
use burn_store::{ModuleSnapshot, PathFilter};

// Direct transfer - all compatible tensors
let snapshots = model1.collect(None, None);
let result = model2.apply(snapshots, None, None);

// Selective transfer with filtering
let filter = PathFilter::new().with_regex(r"^encoder\..*");
let snapshots = model1.collect(Some(filter.clone()), None);
let result = model2.apply(snapshots, Some(filter), None);

// Transfer with path transformation
let mut snapshots = model1.collect(None, None);
for snapshot in &mut snapshots {
    snapshot.full_path = snapshot.full_path.replace("encoder.", "transformer.encoder.");
}
model2.apply(snapshots, None, None);
```

#### Partial Loading and Exports

```rust
// Export only specific layers
let mut store = SafetensorsStore::from_file("encoder_only.safetensors")
    .with_regex(r"^encoder\..*");
model.collect_to(&mut store)?;

// Load with missing tensors allowed
let mut store = SafetensorsStore::from_file("pretrained.safetensors")
    .allow_partial(true);
let result = model.apply_from(&mut store)?;
println!("Loaded: {}, Missing: {:?}", result.applied.len(), result.missing);
```

#### Merging Multiple Models

```rust
// Merge weights from different sources
let mut merged = Vec::new();
merged.extend(base_model.collect(None, None));

// Add encoder from specialized model
let encoder_filter = PathFilter::new().with_regex(r"^encoder\..*");
merged.extend(specialized_model.collect(Some(encoder_filter), None));

// Apply merged weights
target_model.apply(merged, None, None);

// Alternative: Sequential loading from files
let mut base_store = SafetensorsStore::from_file("base.safetensors");
model.apply_from(&mut base_store)?;

let mut encoder_store = SafetensorsStore::from_file("encoder.safetensors")
    .with_regex(r"^encoder\..*")
    .allow_partial(true);
model.apply_from(&mut encoder_store)?;  // Overlays encoder weights
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

### Loading Benchmarks

Compares 6 loading methods: BurnpackStore, NamedMpkFileRecorder, SafetensorsStore,
SafetensorsFileRecorder, PytorchStore, and PyTorchFileRecorder.

```bash
# Generate model files first (one-time setup)
cd crates/burn-store
uv run benches/generate_unified_models.py

# Run unified loading benchmark with default backend (NdArray CPU)
cargo bench --bench unified_loading

# Run with specific backend
cargo bench --bench unified_loading --features metal    # Apple GPU
cargo bench --bench unified_loading --features wgpu     # WebGPU
cargo bench --bench unified_loading --features cuda     # NVIDIA GPU
cargo bench --bench unified_loading --features candle   # Candle backend
cargo bench --bench unified_loading --features tch      # LibTorch

# Run with multiple backends
cargo bench --bench unified_loading --features wgpu,tch
```

### Saving Benchmarks

Compares 3 saving methods: BurnpackStore, NamedMpkFileRecorder, and SafetensorsStore.

```bash
# Run unified saving benchmark with default backend (NdArray CPU)
cargo bench --bench unified_saving

# Run with specific backend
cargo bench --bench unified_saving --features metal    # Apple GPU
cargo bench --bench unified_saving --features wgpu     # WebGPU
cargo bench --bench unified_saving --features cuda     # NVIDIA GPU
cargo bench --bench unified_saving --features candle   # Candle backend
cargo bench --bench unified_saving --features tch      # LibTorch

# Run with multiple backends
cargo bench --bench unified_saving --features wgpu,tch
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

This project is dual-licensed under MIT and Apache-2.0.
