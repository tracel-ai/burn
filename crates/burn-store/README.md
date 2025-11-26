# Burn Store

> Advanced model storage and serialization for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-store.svg)](https://crates.io/crates/burn-store)

A comprehensive storage library for Burn that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

## Features

### Core Capabilities

- **Burnpack Format** - Native Burn format with CBOR metadata, memory-mapped loading, ParamId
  persistence for stateful training, and no-std support
- **SafeTensors Format** - Industry-standard format for secure and efficient tensor serialization
- **PyTorch Support** - Direct loading of PyTorch .pth/.pt files with automatic weight
  transformation
- **Zero-Copy Loading** - Memory-mapped files and lazy tensor materialization for optimal
  performance
- **Cross-Framework Support** - Seamless PyTorch ↔ Burn model conversion with automatic adaptations
- **Flexible Filtering** - Load/save specific model subsets with regex, exact paths, or custom
  predicates
- **Tensor Remapping** - Rename tensors during load/save for framework compatibility
- **No-std Support** - Burnpack and SafeTensors formats available in embedded and WASM
  environments

Note: no-std support for SafeTensors format is temporarily disabled due to
https://github.com/huggingface/safetensors/issues/650 not released yet.

### Advanced Features

- **Framework Adapters** - Automatic weight transposition and parameter renaming for PyTorch
  compatibility
- **Lazy Transformations** - Chain tensor transformations without materializing intermediate data
- **Partial Loading** - Continue loading even when some tensors are missing
- **Custom Metadata** - Attach version info, training details, or other metadata to saved models

## Quick Start

### Basic Save and Load

#### Burnpack (Native Format)

```rust
use burn_store::{ModuleSnapshot, BurnpackStore};

// Save a model with metadata
let mut store = BurnpackStore::from_file("model.bpk")
    .metadata("version", "1.0")
    .metadata("description", "My trained model");
model.save_into(&mut store)?;

// Load a model (automatically memory-mapped when available)
let mut store = BurnpackStore::from_file("model.bpk");
model.load_from(&mut store)?;
```

**Performance**: Burnpack provides faster loading times and reduced memory overhead compared to
other formats.

**Training State Persistence**: Burnpack automatically preserves parameter identifiers (ParamId) for
stateful training continuation.

#### SafeTensors

```rust
use burn_store::{ModuleSnapshot, SafetensorsStore};

// Save a model
let mut store = SafetensorsStore::from_file("model.safetensors");
model.save_into(&mut store)?;

// Load a model
let mut store = SafetensorsStore::from_file("model.safetensors");
model.load_from(&mut store)?;
```

### Filtering Tensors

```rust
// Save only encoder layers
let mut store = SafetensorsStore::from_file("encoder.safetensors")
    .with_regex(r"^encoder\..*")
    .metadata("subset", "encoder_only");

model.save_into(&mut store)?;

// Load with multiple filter patterns (OR logic)
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")      // Include encoder tensors
    .with_regex(r".*\.bias$")          // OR include any bias tensors
    .with_full_path("decoder.scale"); // OR include specific tensor

model.load_from(&mut store)?;
```

### PyTorch Interoperability

```rust
use burn_store::{PyTorchToBurnAdapter, BurnToPyTorchAdapter, PytorchStore};

// Load PyTorch .pth file directly (PyTorchToBurnAdapter is applied automatically)
// Note: skip_enum_variants defaults to true for PytorchStore
let mut store = PytorchStore::from_file("pytorch_model.pth")
    .with_top_level_key("state_dict")         // Access nested state dict
    .allow_partial(true);                     // Skip unknown tensors

burn_model.load_from(&mut store)?;

// Load PyTorch model from SafeTensors
let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter)  // Auto-transpose linear weights
    .skip_enum_variants(true)                 // Handle enum variant name differences
    .allow_partial(true);                     // Skip unknown PyTorch tensors

burn_model.load_from(&mut store)?;

// Save Burn model for PyTorch (with enum variant skipping)
let mut store = SafetensorsStore::from_file("for_pytorch.safetensors")
    .with_to_adapter(BurnToPyTorchAdapter)    // Convert back to PyTorch format
    .skip_enum_variants(true);                // Omit enum variants for PyTorch compatibility

burn_model.save_into(&mut store)?;
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
// Burnpack: Save to memory buffer
let mut store = BurnpackStore::from_bytes(None)
    .with_regex(r"^encoder\..*")
    .metadata("subset", "encoder_only");
model.save_into(&mut store)?;
let bytes = store.get_bytes()?;

// Burnpack: Load from memory buffer (no-std compatible)
let mut store = BurnpackStore::from_bytes(Some(bytes))
    .allow_partial(true);
let result = model.load_from(&mut store)?;

// SafeTensors: Memory operations
let mut store = SafetensorsStore::from_bytes(None)
    .with_regex(r"^encoder\..*");
model.save_into(&mut store)?;
let bytes = store.get_bytes()?;

println!("Loaded {} tensors", result.applied.len());
if !result.missing.is_empty() {
    println!("Missing: {:?}", result.missing);
}
```

Both BurnpackStore and SafetensorsStore support no-std environments when using byte operations

### Model Surgery and Partial Operations

Burn Store enables sophisticated model surgery operations for selectively loading, saving, and
transferring parts of models.

#### Direct Model-to-Model Transfer

```rust
use burn_store::{ModuleSnapshot, PathFilter};

// Direct transfer - all compatible tensors
let snapshots = model1.collect(None, None, false);
let result = model2.apply(snapshots, None, None, false);

// Selective transfer with filtering
let filter = PathFilter::new().with_regex(r"^encoder\..*");
let snapshots = model1.collect(Some(filter.clone()), None, false);
let result = model2.apply(snapshots, Some(filter), None, false);

// Transfer with path transformation
let mut snapshots = model1.collect(None, None, false);
for snapshot in &mut snapshots {
    snapshot.full_path = snapshot.full_path.replace("encoder.", "transformer.encoder.");
}
model2.apply(snapshots, None, None, false);
```

#### Partial Loading and Exports

```rust
// Export only specific layers
let mut store = SafetensorsStore::from_file("encoder_only.safetensors")
    .with_regex(r"^encoder\..*");
model.save_into(&mut store)?;

// Load with missing tensors allowed
let mut store = SafetensorsStore::from_file("pretrained.safetensors")
    .allow_partial(true);
let result = model.load_from(&mut store)?;
println!("Loaded: {}, Missing: {:?}", result.applied.len(), result.missing);
```

#### Merging Multiple Models

```rust
// Merge weights from different sources
let mut merged = Vec::new();
merged.extend(base_model.collect(None, None, false));

// Add encoder from specialized model
let encoder_filter = PathFilter::new().with_regex(r"^encoder\..*");
merged.extend(specialized_model.collect(Some(encoder_filter), None, false));

// Apply merged weights
target_model.apply(merged, None, None, false);

// Alternative: Sequential loading from files
let mut base_store = SafetensorsStore::from_file("base.safetensors");
model.load_from(&mut base_store)?;

let mut encoder_store = SafetensorsStore::from_file("encoder.safetensors")
    .with_regex(r"^encoder\..*")
    .allow_partial(true);
model.load_from(&mut encoder_store)?;  // Overlays encoder weights
```

### Complete Example: Migrating PyTorch Models

```rust
use burn_store::{ModuleSnapshot, PytorchStore};

// Load directly from PyTorch .pth file (automatic PyTorchToBurnAdapter)
// Note: skip_enum_variants defaults to true for PytorchStore
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
let result = model.load_from(&mut store)?;

println!("Successfully migrated {} tensors", result.applied.len());
if !result.errors.is_empty() {
    println!("Errors: {:?}", result.errors);
}

// Save the migrated model in SafeTensors format
let mut save_store = SafetensorsStore::from_file("migrated_model.safetensors")
    .metadata("source", "pytorch")
    .metadata("converted_by", "burn-store");

model.save_into(&mut save_store)?;
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
let result = model.load_from(&mut store)?;

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

- `metadata(key, value)` - Add custom metadata (Burnpack and SafeTensors)
- `allow_partial(bool)` - Continue on missing tensors
- `validate(bool)` - Toggle validation
- `skip_enum_variants(bool)` - Skip enum variant names in paths for PyTorch compatibility
- `with_top_level_key(key)` - Access nested dict in PyTorch files
- `overwrite(bool)` - Allow overwriting existing files (Burnpack)

### Inspecting Burnpack Files

Generate and examine a sample file:

```bash
cargo run --example burnpack_inspect sample.bpk
hexdump -C sample.bpk | head -20
```

The example creates a sample model and outputs inspection commands for examining the binary format.

## License

This project is dual-licensed under MIT and Apache-2.0.
