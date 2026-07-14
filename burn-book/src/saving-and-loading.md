# Saving and Loading Models

Saving your trained machine learning model is quite easy. As mentioned in the
[Record](./building-blocks/record.md) section, a module's parameters are captured in a
`ModuleRecord` and serialized to the [burnpack](./building-blocks/record.md) format (`.bpk`).

```rust, ignore
use burn::store::ModuleRecord;

// Take a record of the model's parameters and save it to disk.
model
    .into_record()
    .save(model_path)
    .expect("Should be able to save the model");
```

Note that the `.bpk` file extension is appended automatically when the path has none, so only the
file path and base name need to be provided.

Now that you have a trained model saved to your disk, you can just as easily load it back.

```rust, ignore
// Load the record from the burnpack file.
let record = ModuleRecord::load(model_path)
    .expect("Should be able to load the model weights from the provided file");

// Apply the loaded weights to a model.
model = model.load_record(record);
```

The record is backend-independent: weights saved with one backend can be loaded on another. If you
need to convert precision, call `model.cast(dtype)` before taking the record, or use
`record.cast_to_module_dtype()` when loading.

## Initialization from Recorded Weights

The most straightforward way to load weights for a module is simply by using
[load_record](https://burn.dev/docs/burn/module/trait.Module.html#method.load_record). Note that
parameter initialization is lazy, therefore no actual tensor allocation and GPU/CPU kernels are
executed before the module is used. This means that you can use `init(device)` followed by
`load_record(record)` without any meaningful performance cost.

```rust, ignore
use burn::store::ModuleRecord;

// Create a dummy initialized model to save.
let device = Default::default();
let model = Model::init(&device);

// Save its parameters to a burnpack file.
model
    .into_record()
    .save(model_path)
    .expect("Should be able to save the model");
```

Afterwards, the model can just as easily be loaded from the record saved on disk.

```rust, ignore
// Load the model record from the burnpack file.
let record = ModuleRecord::load(model_path)
    .expect("Could not load model weights");

// Initialize a new model with the loaded record/weights.
let model = Model::init(&device).load_record(record);
```

For partial loading (only some parameters present in the record), use
`record.allow_partial(true)` before applying it, or `model.try_load_record(record)` for fallible
loading.

## Model Weight Store

While the `ModuleRecord` API above works well for basic saving and loading, the `burn-store` crate
adds memory efficiency and flexibility on top of the same burnpack format. It provides zero-copy
memory-mapped loading, cross-framework interoperability (PyTorch and SafeTensors), key remapping,
partial loading, and filtering.

### Supported Formats

| Format          | Extension      | Description                                                                               |
| --------------- | -------------- | ----------------------------------------------------------------------------------------- |
| **Burnpack**    | `.bpk`         | Burn's native format with fast loading, zero-copy support, and training state persistence |
| **SafeTensors** | `.safetensors` | Industry-standard format from Hugging Face for secure tensor serialization                |
| **PyTorch**     | `.pt`, `.pth`  | Direct loading of PyTorch model weights (read-only)                                       |

### Saving a Model

```rust, ignore
use burn_store::{ModuleSnapshot, BurnpackStore};

// Save to Burnpack (recommended)
let mut store = BurnpackStore::from_file("model.bpk");
model.save_into(&mut store)?;

// Or save to SafeTensors
use burn_store::SafetensorsStore;
let mut store = SafetensorsStore::from_file("model.safetensors");
model.save_into(&mut store)?;
```

### Loading a Model

```rust, ignore
use burn_store::{ModuleSnapshot, BurnpackStore};

let device = Default::default();
let mut model = MyModel::init(&device);

// Load from Burnpack
let mut store = BurnpackStore::from_file("model.bpk");
model.load_from(&mut store)?;
```

### Loading from PyTorch

You can load weights directly from PyTorch `.pt` files:

```rust, ignore
use burn_store::{ModuleSnapshot, PytorchStore};

let mut model = MyModel::init(&device);
let mut store = PytorchStore::from_file("pytorch_model.pt");
model.load_from(&mut store)?;
```

#### Exporting from PyTorch

Save only the model weights (state_dict), not the entire model:

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, (2, 2))
        self.conv2 = nn.Conv2d(2, 2, (2, 2), bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))

model = Net()
torch.save(model.state_dict(), "model.pt")  # Correct: save state_dict
# torch.save(model, "model.pt")             # Wrong: saves entire model
```

#### Accessing Nested State Dicts

Some PyTorch checkpoints nest the state_dict under a key:

```rust, ignore
let mut store = PytorchStore::from_file("checkpoint.pt")
    .with_top_level_key("state_dict");
model.load_from(&mut store)?;
```

### Loading from SafeTensors

For SafeTensors files exported from PyTorch, use the adapter for proper weight transformation:

```rust, ignore
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

let mut model = MyModel::init(&device);
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_from_adapter(PyTorchToBurnAdapter);
model.load_from(&mut store)?;
```

For SafeTensors files created by Burn, no adapter is needed:

```rust, ignore
let mut store = SafetensorsStore::from_file("model.safetensors");
model.load_from(&mut store)?;
```

#### Exporting from PyTorch to SafeTensors

```python
from safetensors.torch import save_file

model = Net()
save_file(model.state_dict(), "model.safetensors")
```

### Saving for PyTorch Compatibility

Use the adapter when saving for PyTorch consumption:

```rust, ignore
use burn_store::{BurnToPyTorchAdapter, SafetensorsStore};

let mut store = SafetensorsStore::from_file("for_pytorch.safetensors")
    .with_to_adapter(BurnToPyTorchAdapter)
    .skip_enum_variants(true);
model.save_into(&mut store)?;
```

### Handling Load Results

The `load_from` method returns detailed information about the loading process.

> **Note:** Inspecting `result.missing`, `result.errors`, etc. requires the store to be configured with
> [`.allow_partial(true)`](#partial-loading). Without it, a missing tensor causes a hard `Err`
> before you ever receive an `ApplyResult`.

```rust, ignore
// Use .allow_partial(true) to get an ApplyResult with structured info
let mut store = PytorchStore::from_file("pretrained.pt")
    .allow_partial(true);

let result = model.load_from(&mut store)?;

// Print a formatted summary with suggestions
println!("{}", result);

// Or inspect individual fields
println!("Applied: {} tensors", result.applied.len());
println!("Missing: {:?}", result.missing);
println!("Errors: {:?}", result.errors);

if result.is_success() {
    println!("All tensors loaded successfully");
}
```

If you don't use `.allow_partial(true)`, use normal error handling:

```rust, ignore
match model.load_from(&mut store) {
    Ok(result) => println!("Loaded successfully: {} tensors", result.applied.len()),
    Err(e) => eprintln!("Failed to load: {e}"),
}
```

### Adding Metadata

Burnpack and SafeTensors support custom metadata:

```rust, ignore
let mut store = BurnpackStore::from_file("model.bpk")
    .metadata("version", "1.0")
    .metadata("description", "My trained model")
    .metadata("epochs", "100");
model.save_into(&mut store)?;
```

### Advanced Features

#### Key Remapping

Remap parameter names using regex patterns when model structures don't match:

```rust, ignore
let mut store = PytorchStore::from_file("model.pt")
    // Remove prefix: "model.conv1.weight" -> "conv1.weight"
    .with_key_remapping(r"^model\.", "")
    // Rename: "layer1" -> "encoder.layer1"
    .with_key_remapping(r"^layer", "encoder.layer");
model.load_from(&mut store)?;
```

For complex remapping:

```rust, ignore
use burn_store::KeyRemapper;

let remapper = KeyRemapper::new()
    .add_pattern(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")?
    .add_pattern(r"\.attn\.", ".attention.")?;

let mut store = SafetensorsStore::from_file("model.safetensors")
    .remap(remapper);
```

#### Partial Loading

Load weights even when some tensors are missing:

```rust, ignore
let mut store = PytorchStore::from_file("pretrained.pt")
    .allow_partial(true);

let result = model.load_from(&mut store)?;
println!("Missing (initialized randomly): {:?}", result.missing);
```

#### Filtering Tensors

Load or save only specific layers:

```rust, ignore
// Load only encoder layers
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")
    .allow_partial(true);

// Save only encoder layers
let mut store = SafetensorsStore::from_file("encoder.safetensors")
    .with_regex(r"^encoder\..*");
model.save_into(&mut store)?;

// Multiple patterns (OR logic)
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_regex(r"^encoder\..*")      // encoder tensors
    .with_regex(r".*\.bias$")          // OR any bias tensors
    .with_full_path("decoder.scale"); // OR specific tensor
```

#### Non-Contiguous Layer Indices

PyTorch `nn.Sequential` with mixed layers creates non-contiguous indices. `PytorchStore`
automatically remaps these:

```
PyTorch: fc.0.weight, fc.2.weight, fc.4.weight  (gaps from ReLU layers)
Burn:    fc.0.weight, fc.1.weight, fc.2.weight  (contiguous)
```

This is enabled by default. Disable if needed:

```rust, ignore
let mut store = PytorchStore::from_file("model.pt")
    .map_indices_contiguous(false);
```

#### Zero-Copy Loading

For embedded models or large files, use zero-copy loading to avoid memory copies:

```rust, ignore
// Embedded model (compile-time)
static MODEL_DATA: &[u8] = include_bytes!("model.bpk");
let mut store = BurnpackStore::from_static(MODEL_DATA);
model.load_from(&mut store)?;

// Large file (memory-mapped)
let mut store = BurnpackStore::from_file("large_model.bpk")
    .zero_copy(true);
model.load_from(&mut store)?;
```

#### Half-Precision Storage

Save models at half precision (F16) to reduce file size by ~50%, then load back at full precision:

```rust, ignore
use burn_store::{ModuleSnapshot, BurnpackStore, HalfPrecisionAdapter};

let adapter = HalfPrecisionAdapter::new();

// Save: F32 -> F16 (same adapter for both directions)
let mut store = BurnpackStore::from_file("model_f16.bpk")
    .with_to_adapter(adapter.clone());
model.save_into(&mut store)?;

// Load: F16 -> F32
let mut store = BurnpackStore::from_file("model_f16.bpk")
    .with_from_adapter(adapter);
model.load_from(&mut store)?;
```

By default, weights in Linear, Embedding, Conv\*, LayerNorm, GroupNorm, InstanceNorm, RmsNorm, and
PRelu modules are converted. BatchNorm is excluded because its running variance can underflow in
F16. Customize with `with_module()` and `without_module()`:

```rust, ignore
// Keep LayerNorm at full precision
let adapter = HalfPrecisionAdapter::new()
    .without_module("LayerNorm");

// Add a custom module to the conversion set
let adapter = HalfPrecisionAdapter::new()
    .with_module("CustomLayer");
```

#### Direct Tensor Access

Inspect tensors without loading into a model:

```rust, ignore
use burn_store::ModuleStore;

let mut store = PytorchStore::from_file("model.pt");

// List all tensor names
let names = store.keys()?;

// Get specific tensor
if let Some(snapshot) = store.get_snapshot("encoder.layer0.weight")? {
    println!("Shape: {:?}, DType: {:?}", snapshot.shape, snapshot.dtype);
}
```

#### Model Surgery

Transfer weights between models:

```rust, ignore
use burn_store::{ModuleSnapshot, PathFilter};

// Transfer all weights
let snapshots = model1.collect(None, None, false);
model2.apply(snapshots, None, None, false);

// Transfer only encoder weights
let filter = PathFilter::new().with_regex(r"^encoder\..*");
let snapshots = model1.collect(Some(filter.clone()), None, false);
model2.apply(snapshots, Some(filter), None, false);
```

### API Reference

#### Builder Methods

| Category      | Method                         | Description                  |
| ------------- | ------------------------------ | ---------------------------- |
| **Filtering** | `with_regex(pattern)`          | Filter by regex pattern      |
|               | `with_full_path(path)`         | Include specific tensor      |
|               | `with_predicate(fn)`           | Custom filter logic          |
| **Remapping** | `with_key_remapping(from, to)` | Regex-based renaming         |
|               | `remap(KeyRemapper)`           | Complex remapping rules      |
| **Adapters**  | `with_from_adapter(adapter)`   | Loading transformations      |
|               | `with_to_adapter(adapter)`     | Saving transformations       |
|               | `HalfPrecisionAdapter::new()`  | F32/F16 mixed-precision      |
| **Config**    | `allow_partial(bool)`          | Continue on missing tensors  |
|               | `with_top_level_key(key)`      | Access nested dict (PyTorch) |
|               | `skip_enum_variants(bool)`     | Skip enum variants in paths  |
|               | `map_indices_contiguous(bool)` | Remap non-contiguous indices |
|               | `metadata(key, value)`         | Add custom metadata          |
|               | `zero_copy(bool)`              | Enable zero-copy loading     |

#### Direct Access Methods

| Method                | Description                      |
| --------------------- | -------------------------------- |
| `keys()`              | Get ordered list of tensor names |
| `get_all_snapshots()` | Get all tensors as BTreeMap      |
| `get_snapshot(name)`  | Get specific tensor by name      |

### Troubleshooting

#### Common Issues

1. **"Missing source values" error**: You saved the entire PyTorch model instead of the state_dict.
   Re-export with `torch.save(model.state_dict(), "model.pt")`.

2. **Shape mismatch**: Your Burn model doesn't match the source architecture. Verify layer
   configurations (channels, kernel sizes, bias settings).

3. **Key not found**: Parameter names don't match. Use `with_key_remapping()` or inspect keys:

   ```rust, ignore
   let store = PytorchStore::from_file("model.pt");
   println!("Available keys: {:?}", store.keys()?);
   ```

#### Inspecting Files

Use [Netron](https://github.com/lutzroeder/netron) to visualize `.pt` and `.safetensors` files.

For Burnpack files:

```bash
cargo run --example burnpack_inspect model.bpk
```
