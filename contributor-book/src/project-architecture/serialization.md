# Serialization

An important aspect of a deep learning framework is the ability to save and load training state to
and from disk. Burn serializes its records with the **burnpack** format, a compact binary container
implemented by the `burn-pack` crate.

## Constraints

1. **Users should be able to add any field to a module, even fields that are not serializable.**

   This can include constants, database connections, other module references, or any other
   information. Only the parameters (tensors) should be serialized; the structure of the module
   itself is encapsulated by its configuration (hyperparameters).

2. **Records should be decoupled from the backend in use.**

   A record holds plain tensor data ([`TensorData`]), so weights saved with one backend can be
   loaded on another. Parameter initialization is lazy, so loading a record does not require
   eagerly materializing the module first.

3. **The format should be fast to load and embeddable.**

   Tensor data is stored contiguously and aligned, so it can be read back with zero-copy /
   memory-mapped loading, and a record can be saved straight to bytes for `no-std` environments.

## The burnpack format

The `burn-pack` crate is intentionally minimal and tensor-library-agnostic: it knows how to read and
write the container format but has no notion of Burn modules. A burnpack file has three parts:

```text
┌────────────────────────────────────────────────────────────┐
│ Header (fixed size)                                          │
│   magic "BURN", format version, metadata byte length         │
├────────────────────────────────────────────────────────────┤
│ Metadata (CBOR)                                              │
│   tensors : map<name, descriptor>                            │
│     dtype, shape, data_offsets, optional param_id            │
│   scalars  : map<name, typed scalar>                         │
│   metadata : map<string, string>  user key/value pairs       │
├────────────────────────────────────────────────────────────┤
│ Tensor data section                                          │
│   each tensor's bytes start on a 256-byte boundary so the    │
│   data can be sliced zero-copy / memory-mapped from a file   │
└────────────────────────────────────────────────────────────┘
```

All multi-byte integers are little-endian. Tensor entries carry an optional `param_id` used to
preserve a parameter's identity across save/load. Besides tensors, a pack can store named **typed
scalars** (integers, floats, booleans), which the optimizer and learning rate scheduler records use
to persist their non-tensor state.

A pack is written with `burn_pack::Writer` and read back with `burn_pack::Reader`, both operating on
`burn_pack::Tensor` entries plus a scalar map.

## The three record types

Higher layers bridge their state to and from burnpack through three record types. Each one can be
serialized to a file (`save` / `load`, appending the `.bpk` extension when the path has none) or to
an in-memory byte buffer (`into_bytes` / `from_bytes`).

### `ModuleRecord` (`burn-core`, `burn::store`)

Holds a module's parameters: a flat list of `(path, ParamId, TensorData)` entries keyed by module
path. It is produced and applied through the [`Module`] trait itself rather than a separate codegen
type:

- `module.into_record()` walks the module with a `ModuleVisitor` (the `Collector`), recording each
  float/int/bool parameter under its dotted path.
- `module.load_record(record)` (or the fallible `try_load_record`) walks the module with a
  `ModuleMapper` that looks each parameter up by path and loads the matching tensor.

Load-time behavior is configured with builder methods on the record, ignored when saving:

- `allow_partial(bool)` — tolerate module parameters absent from the record.
- `validate(bool)` — toggle shape-mismatch / missing-tensor validation.
- `with_dtype_policy(..)` / `cast_to_module_dtype()` — choose whether a parameter adopts the
  record's dtype (`DTypePolicy::FromRecord`, the default) or casts the data to the module
  parameter's current dtype (`DTypePolicy::CastToModule`).

The save-side dtype is not configurable: the record stores whatever dtype the module currently
holds. The dtype applied on load is controlled by the record's `DTypePolicy`
(`.cast_to_module_dtype()` / `.with_dtype_policy(..)`).

This module in `burn-core` is intentionally tiny — no filtering, key remapping, or adapters. The
richer snapshot/import tooling (filtering, key remapping, PyTorch/SafeTensors cross-framework stores)
lives in the `burn-store` crate.

### `OptimizerRecord` (`burn-optim`)

Holds an optimizer's per-parameter state. Unlike a module record (keyed by module path), it is keyed
per parameter: each parameter's state is decomposed into tensors named `"{param_id}.{field}"` plus a
few typed scalar entries kept in the burnpack scalar map (including a `__rank` scalar so the state
can be reconstructed without inferring rank from tensor shapes).

- `optimizer.to_record()` flattens each parameter's `DynState` into tensors and scalars.
- `optimizer.load_record(record)` reconstructs the states (no device argument: tensors load on the
  default device and migrate to each parameter's device on the next step).

### `LrSchedulerRecord` (`burn-optim`)

Holds a learning rate scheduler's state, which is just a handful of scalars (step counters, current
learning rate) and no tensors. Produced/applied through the [`LrScheduler`] trait's `to_record()` /
`load_record()`. Composed schedulers nest their children's records under an index prefix
(`with_record` / `record`).

## Checkpointing in `burn-train`

During training, `burn-train` defines a `Checkpoint` trait (`save(path)` / `load(path)`) implemented
for all three record types — `ModuleRecord`, `OptimizerRecord`, `LrSchedulerRecord` — and for `()`
(a stateless no-op). The `Checkpointer<R: Checkpoint>` trait drives periodic saves; the
`FileCheckpointer` writes each record to `{name}-{epoch}.bpk` under the experiment directory. This is
how the model, optimizer, and scheduler are persisted and restored across epochs.

## Notes

- There is no `Recorder`, `PrecisionSettings`, `Module::Record` associated type, or `#[derive(Record)]`
  any more. All of those were part of the previous serde-based record system, which has been removed.
- Cross-framework import/export (PyTorch `.pt`, SafeTensors) still lives in `burn-store`
  (`PytorchStore`, `SafetensorsStore`, `BurnpackStore`).

[`TensorData`]: https://burn.dev/docs/burn/tensor/struct.TensorData.html
[`Module`]: https://burn.dev/docs/burn/module/trait.Module.html
[`LrScheduler`]: https://burn.dev/docs/burn/lr_scheduler/trait.LrScheduler.html
