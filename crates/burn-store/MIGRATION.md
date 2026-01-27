# Migration Guide: burn-import to burn-store

This guide helps you migrate from the deprecated `burn-import` recorders (`PyTorchFileRecorder`,
`SafetensorsFileRecorder`) to the new `burn-store` API (`PytorchStore`, `SafetensorsStore`).

## Overview

The new `burn-store` API provides:

- **Simpler API**: Load directly into models instead of records
- **Fluent builder pattern**: Chain configuration methods
- **Better error handling**: Detailed load results with applied/missing/errors info
- **Bidirectional support**: Both load and save operations
- **More features**: Filtering, partial loading, metadata, zero-copy loading

## Quick Migration

### PyTorch Files (.pt/.pth)

**Before (burn-import):**

```rust
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

// Load into a record, then create model from record
let record: ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load("model.pt".into(), &device)
    .expect("Failed to load");

let model = Model::init(&device).load_record(record);
```

**After (burn-store):**

```rust
use burn_store::{ModuleSnapshot, PytorchStore};

// Initialize model, then load weights directly
let mut model = Model::init(&device);
let mut store = PytorchStore::from_file("model.pt");
model.load_from(&mut store).expect("Failed to load");
```

### SafeTensors Files (.safetensors)

**Before (burn-import):**

```rust
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

let record: ModelRecord<B> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load("model.safetensors".into(), &device)
    .expect("Failed to load");

let model = Model::init(&device).load_record(record);
```

**After (burn-store):**

```rust
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

let mut model = Model::init(&device);

// For SafeTensors exported from PyTorch, use the adapter
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter);
model.load_from(&mut store).expect("Failed to load");

// For native Burn SafeTensors, no adapter needed
let mut store = SafetensorsStore::from_file("model.safetensors");
model.load_from(&mut store).expect("Failed to load");
```

## API Mapping

### PyTorchFileRecorder Options

| burn-import                                    | burn-store                                  |
| ---------------------------------------------- | ------------------------------------------- |
| `LoadArgs::new(path)`                          | `PytorchStore::from_file(path)`             |
| `.with_key_remap(pattern, replacement)`        | `.with_key_remapping(pattern, replacement)` |
| `.with_top_level_key(key)`                     | `.with_top_level_key(key)`                  |
| `.with_debug_print()`                          | _(use tracing/logging instead)_             |
| `PyTorchFileRecorder::<FullPrecisionSettings>` | _(precision handled automatically)_         |

### SafetensorsFileRecorder Options

| burn-import                                        | burn-store                                  |
| -------------------------------------------------- | ------------------------------------------- |
| `LoadArgs::new(path)`                              | `SafetensorsStore::from_file(path)`         |
| `.with_key_remap(pattern, replacement)`            | `.with_key_remapping(pattern, replacement)` |
| `.with_adapter_type(AdapterType::PyTorch)`         | `.with_from_adapter(PyTorchToBurnAdapter)`  |
| `.with_adapter_type(AdapterType::NoAdapter)`       | _(default, no adapter)_                     |
| `.with_debug_print()`                              | _(use tracing/logging instead)_             |
| `SafetensorsFileRecorder::<FullPrecisionSettings>` | _(precision handled automatically)_         |

## Detailed Examples

### Key Remapping

**Before:**

```rust
let args = LoadArgs::new("model.pt".into())
    .with_key_remap("conv\\.(.*)", "$1")
    .with_key_remap("^old_prefix\\.", "new_prefix.");

let record: ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load(args, &device)?;
```

**After:**

```rust
let mut store = PytorchStore::from_file("model.pt")
    .with_key_remapping("conv\\.(.*)", "$1")
    .with_key_remapping("^old_prefix\\.", "new_prefix.");

model.load_from(&mut store)?;
```

### Top-Level Key Access

**Before:**

```rust
let args = LoadArgs::new("checkpoint.pt".into())
    .with_top_level_key("state_dict");

let record: ModelRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    .load(args, &device)?;
```

**After:**

```rust
let mut store = PytorchStore::from_file("checkpoint.pt")
    .with_top_level_key("state_dict");

model.load_from(&mut store)?;
```

### PyTorch Adapter for SafeTensors

**Before:**

```rust
use burn_import::safetensors::{AdapterType, LoadArgs};

let args = LoadArgs::new("pytorch_model.safetensors".into())
    .with_adapter_type(AdapterType::PyTorch);

let record: ModelRecord<B> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(args, &device)?;
```

**After:**

```rust
use burn_store::{PyTorchToBurnAdapter, SafetensorsStore};

let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter);

model.load_from(&mut store)?;
```

## New Features in burn-store

### Partial Loading

Handle missing tensors gracefully:

```rust
let mut store = PytorchStore::from_file("model.pt")
    .allow_partial(true);

let result = model.load_from(&mut store)?;
println!("Loaded: {:?}", result.applied);
println!("Missing: {:?}", result.missing);
```

### Filtering

Load only specific tensors:

```rust
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")  // Only encoder layers
    .allow_partial(true);

model.load_from(&mut store)?;
```

### Saving Models

Save models (not supported by old recorders):

```rust
// Save to SafeTensors
let mut store = SafetensorsStore::from_file("output.safetensors")
    .metadata("version", "1.0");
model.save_into(&mut store)?;

// Save to Burnpack (native format)
let mut store = BurnpackStore::from_file("output.bpk");
model.save_into(&mut store)?;
```

### Load Results

Get detailed information about loading:

```rust
let result = model.load_from(&mut store)?;

// Print the full result for debugging - shows applied, skipped, missing, and errors
println!("{}", result);

// Or access individual fields
println!("Applied: {} tensors", result.applied.len());
println!("Skipped: {} tensors", result.skipped.len());
println!("Missing: {:?}", result.missing);
println!("Errors: {:?}", result.errors);

// Check if fully successful
if result.is_success() {
    println!("All tensors loaded successfully");
}
```

The `LoadResult` implements `Display`, so printing it shows a formatted summary with suggestions for
common issues (e.g., using `allow_partial(true)` for missing tensors).

## Updating Cargo.toml

**Before:**

```toml
[dependencies]
burn-import = { version = "0.x", features = ["pytorch", "safetensors"] }
```

**After:**

```toml
[dependencies]
burn-store = { version = "0.x", features = ["pytorch", "safetensors"] }
```

## Common Migration Issues

### 1. Model vs Record

The new API loads directly into models, not records. Update your model initialization:

```rust
// Before: Create record, then model from record
let record = recorder.load(...)?;
let model = Model::init(&device).load_record(record);

// After: Create model, then load into it
let mut model = Model::init(&device);
model.load_from(&mut store)?;
```

### 2. Inference Functions

If you had functions that took `ModelRecord`, update them to take `Model`:

```rust
// Before
fn infer(record: ModelRecord<B>) {
    let model = Model::init(&device).load_record(record);
    // ...
}

// After
fn infer(model: Model<B>) {
    // Model already has weights loaded
    // ...
}
```

### 3. Precision Settings

The old API required explicit precision settings. The new API handles this automatically:

```rust
// Before: Had to specify FullPrecisionSettings or HalfPrecisionSettings
PyTorchFileRecorder::<FullPrecisionSettings>::default()

// After: Precision handled automatically based on tensor dtype
PytorchStore::from_file("model.pt")
```

### 4. Error Handling

The new API provides richer error information:

```rust
// Before: Simple Result
let record = recorder.load(args, &device)?;

// After: LoadResult with detailed info
let result = model.load_from(&mut store)?;

// Print the result to see a helpful summary with suggestions
println!("{}", result);

// Or handle specific issues programmatically
if !result.errors.is_empty() {
    for (path, error) in &result.errors {
        eprintln!("Error loading {}: {}", path, error);
    }
}
```

## See Also

- [burn-store README](README.md) - Full documentation
- [import-model-weights example](../../examples/import-model-weights/) - Working example
