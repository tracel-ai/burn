# Record

Records are how training state is saved and loaded with Burn. A record holds plain tensor data
(decoupled from the backend in use), so weights saved with one backend can be loaded on another, and
parameter initialization stays lazy.

All records serialize to the **burnpack** format (`.bpk`), Burn's compact binary container
implemented by the `burn-pack` crate.

## The burnpack format

A burnpack file has three parts:

- a small fixed-size **header** (a `"BURN"` magic number, a format version, and the metadata length);
- a **metadata** blob (CBOR) describing each tensor (name, dtype, shape, data offsets, optional
  parameter id), any named **typed scalars**, and user key/value pairs;
- a **tensor data section** where each tensor's bytes start on a 256-byte boundary, so the data can
  be read back with zero-copy / memory-mapped loading.

Storing typed scalars (integers, floats, booleans) alongside tensors is what lets the optimizer and
learning rate scheduler persist their non-tensor state in the same format.

## The three record types

| Record               | Holds                          | Produced from                                  |
| -------------------- | ------------------------------ | ---------------------------------------------- |
| `ModuleRecord`       | a module's parameters          | `module.into_record()`                         |
| `OptimizerRecord`    | the optimizer state            | `optimizer.to_record()`                        |
| `LrSchedulerRecord`  | the learning rate scheduler    | `scheduler.to_record()`                        |

Each record can be written to a file (`save` / `load`, which appends the `.bpk` extension when the
path has none) or to an in-memory byte buffer (`into_bytes` / `from_bytes`, useful for `no-std`
deployment where the bytes are embedded with the compiled code).

### `ModuleRecord`

`ModuleRecord` (in `burn::store`) holds a module's parameters keyed by their path within the module.
It is produced and applied through the `Module` trait itself:

```rust, ignore
use burn::store::ModuleRecord;

// Take a record and save it.
model.into_record().save("model")?; // writes model.bpk

// Load it back and apply it to an initialized module.
let record = ModuleRecord::load("model")?;
let model = ModelConfig::new().init(&device).load_record(record);
```

Load-time behavior is configured with builder methods on the record (ignored when saving):

- `.allow_partial(true)` — load even when some module parameters are absent from the record;
- `.validate(false)` — skip shape-mismatch / missing-tensor validation;
- `.cast_to_module_dtype()` / `.with_dtype_policy(..)` — cast the record's data to the module
  parameter dtypes on load (by default the parameter adopts the record's dtype).

The save-side dtype is not configurable: the record stores whatever dtype the module currently holds.
To control the dtype applied on load, use `.cast_to_module_dtype()` / `.with_dtype_policy(..)` above.
Use `try_load_record` for the fallible variant of `load_record`.

### `OptimizerRecord` and `LrSchedulerRecord`

The optimizer and learning rate scheduler expose the same shape of API, used to checkpoint and resume
training:

```rust, ignore
// Optimizer state (no device needed on load; state migrates to each parameter's device on the
// next step).
optimizer.save("optim")?;
let optimizer = optimizer.load("optim")?;

// Learning rate scheduler state (scalars only).
scheduler.to_record().save("scheduler")?;
let scheduler = scheduler.load_record(LrSchedulerRecord::load("scheduler")?);
```

When training with the `Learner`, these records are saved and restored for you by the checkpointer —
see [Learner](./learner.md).

## Cross-framework formats

To import weights from other ecosystems (PyTorch `.pt`, SafeTensors) or to use the more advanced
store features (key remapping, filtering, half-precision storage), use the `burn-store` crate. See
[Saving and Loading Models](../saving-and-loading.md) for examples.
