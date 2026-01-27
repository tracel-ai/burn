# Burn Store

> Advanced model storage and serialization for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-store.svg)](https://crates.io/crates/burn-store)

A comprehensive storage library for Burn that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

> **Migrating from burn-import?** See the [Migration Guide](MIGRATION.md) for help moving from
> `PyTorchFileRecorder`/`SafetensorsFileRecorder` to the new Store API.

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
- **No-std Support** - Burnpack and SafeTensors formats available in embedded and WASM environments

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

### Contiguous Layer Index Mapping

When loading PyTorch models that use `nn.Sequential` with mixed layer types (e.g., Conv2d + ReLU),
the layer indices may be non-contiguous because only some layers have parameters:

```python
# PyTorch model with non-contiguous indices
self.fc = nn.Sequential(
    nn.Conv2d(...),  # fc.0.weight, fc.0.bias
    nn.ReLU(),       # No parameters (index 1 skipped)
    nn.Conv2d(...),  # fc.2.weight, fc.2.bias
    nn.ReLU(),       # No parameters (index 3 skipped)
    nn.Conv2d(...),  # fc.4.weight, fc.4.bias
)
```

Burn models typically expect contiguous indices (`fc.0`, `fc.1`, `fc.2`). The `map_indices_contiguous`
feature automatically maps non-contiguous indices to contiguous ones:

```rust
// PytorchStore: map_indices_contiguous is ON by default
let mut store = PytorchStore::from_file("model.pth");
// fc.0 -> fc.0, fc.2 -> fc.1, fc.4 -> fc.2

// Disable if your model already has contiguous indices
let mut store = PytorchStore::from_file("model.pth")
    .map_indices_contiguous(false);

// SafetensorsStore: map_indices_contiguous is OFF by default
let mut store = SafetensorsStore::from_file("model.safetensors")
    .map_indices_contiguous(true);  // Enable for PyTorch-exported safetensors
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

### Zero-Copy Loading

For embedded models and large model files, zero-copy loading avoids unnecessary memory allocations
by directly referencing the source data instead of copying it.

#### Embedded Models (Static Data)

```rust
use burn_store::{ModuleSnapshot, BurnpackStore};

// Embed model weights in the binary at compile time
static MODEL_DATA: &[u8] = include_bytes!("model.bpk");

// Zero-copy loading - data stays in binary's .rodata section
let mut store = BurnpackStore::from_static(MODEL_DATA);
model.load_from(&mut store)?;
```

The `from_static()` constructor automatically enables zero-copy mode. Tensor data is sliced directly
from the embedded bytes without heap allocation.

#### File-Based Zero-Copy

```rust
// Memory-mapped file with zero-copy tensor slicing
let mut store = BurnpackStore::from_file("large_model.bpk")
    .zero_copy(true);  // Enable zero-copy slicing
model.load_from(&mut store)?;
```

When `zero_copy(true)` is set, the memory-mapped file is wrapped in `bytes::Bytes` via
`from_owner()`, enabling O(1) slicing operations.

#### In-Memory Zero-Copy

```rust
use burn_tensor::{AllocationProperty, Bytes};

// Create shared bytes for zero-copy
let data: Vec<u8> = load_model_bytes();
let shared = bytes::Bytes::from(data);
let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

// Load with zero-copy enabled
let mut store = BurnpackStore::from_bytes(Some(bytes))
    .zero_copy(true);
model.load_from(&mut store)?;
```

#### When to Use Zero-Copy

| Scenario                            | Recommendation                     |
| ----------------------------------- | ---------------------------------- |
| Embedded models (`include_bytes!`)  | Use `from_static()` (auto-enabled) |
| Large model files                   | Use `from_file().zero_copy(true)`  |
| Repeated loading from same bytes    | Use `from_bytes().zero_copy(true)` |
| One-time load, release memory after | Use default (copy mode)            |

**Note**: Zero-copy keeps the source data alive as long as any tensor references it. Use copy mode
(default) if you need to release the source file/memory immediately after loading.

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

### Direct Tensor Access

All stores provide methods to directly access tensor snapshots without loading into a model. This is
useful for inspection, debugging, selective processing, or building custom loading pipelines.

```rust
use burn_store::{ModuleStore, BurnpackStore, SafetensorsStore, PytorchStore};

// Works with any store type
let mut store = BurnpackStore::from_file("model.bpk");
// let mut store = SafetensorsStore::from_file("model.safetensors");
// let mut store = PytorchStore::from_file("model.pth");

// List all tensor names (ordered)
let names = store.keys()?;
println!("Model contains {} tensors:", names.len());
for name in &names {
    println!("  - {}", name);
}

// Get all tensors as a BTreeMap (cached for repeated access)
let snapshots = store.get_all_snapshots()?;
for (name, snapshot) in snapshots {
    println!("{}: {:?} {:?}", name, snapshot.shape, snapshot.dtype);
}

// Get a specific tensor by name
if let Some(snapshot) = store.get_snapshot("encoder.layer0.weight")? {
    // Lazy loading - data is only fetched when to_data() is called
    let data = snapshot.to_data()?;
    println!("Shape: {:?}, DType: {:?}", data.shape, data.dtype);
}
```

#### Use Cases

- **Model Inspection**: Examine tensor shapes, dtypes, and names without full model instantiation
- **Selective Loading**: Build custom pipelines that only load specific tensors
- **Debugging**: Verify tensor values and compare across different model files
- **Format Conversion**: Read tensors from one format and write to another

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
- `map_indices_contiguous(bool)` - Map non-contiguous layer indices to contiguous (default: `true`
  for PyTorch, `false` for SafeTensors)
- `with_top_level_key(key)` - Access nested dict in PyTorch files
- `overwrite(bool)` - Allow overwriting existing files (Burnpack)
- `zero_copy(bool)` - Enable zero-copy tensor slicing (Burnpack)

#### Direct Tensor Access

- `keys()` - Get ordered list of all tensor names
- `get_all_snapshots()` - Get all tensors as a BTreeMap (cached)
- `get_snapshot(name)` - Get a specific tensor by name

### Inspecting Burnpack Files

Generate and examine a sample file:

```bash
cargo run --example burnpack_inspect sample.bpk
hexdump -C sample.bpk | head -20
```

The example creates a sample model and outputs inspection commands for examining the binary format.

## License

This project is dual-licensed under MIT and Apache-2.0.
