# Migrating to Burn 0.22

Burn 0.22 selects backends at runtime through `Device`. Tensor operations follow Tensor → Bridge →
Dispatch → Backend; ordinary models and tensor functions no longer carry a backend type parameter.
Backend implementations and extensions still use low-level backend traits.

## Cargo features and toolchain

The minimum supported Rust version is 1.95. NdArray and LibTorch are deprecated, the Candle backend
is removed, and backend tracing requires the opt-in `tracing` feature. Enable a feature for each
backend constructor you use; `burn` has no default execution backend.

| 0.21                                                                          | 0.22                                                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `server`                                                                      | `remote-server`                                                                                                                                        |
| `remote` (WebSocket)                                                          | `remote` (Iroh); add `remote-websocket` for WebSocket                                                                                                  |
| `sqlite` through `dataset` or `train`                                         | `sqlite`, enabled explicitly (`sqlite-bundled` is an alias)                                                                                            |
| Linear algebra in `burn::tensor::linalg`                                      | `linalg` feature; `burn::linalg` extension traits                                                                                                      |
| Signal processing in `burn::tensor::signal`                                   | `signal` feature; `burn::signal` or `burn::tensor::signal`. See [Signal Processing Functions](./building-blocks/tensor.md#signal-processing-functions) |
| `burn-store` dependency for SafeTensors and PyTorch                           | `safetensors` and `pytorch` features (imply `store`)                                                                                                   |
| `candle`, `candle-cuda`, `candle-metal`                                       | Removed                                                                                                                                                |
| `router`, `dispatch`, `distributed`, `collective`, `record-item-custom-serde` | Removed; runtime dispatch and `burn::tensor::distributed` are built in                                                                                 |

With `default-features = false`, enable `optim` explicitly if you use `burn::optim` or
`burn::lr_scheduler`; `train` also enables it.

Reinforcement learning is opt-in. Enable the `rl` feature on `burn` to use `burn::rl` and the RL
learner in `burn::train`. If you depend on `burn-train` directly, `rl` is no longer one of its
default features.

## Types and devices

| Previous API                                        | 0.22 API                                                                          |
| --------------------------------------------------- | --------------------------------------------------------------------------------- |
| `Tensor<B, D>` / `Tensor<B, D, Int>`                | `Tensor<D>` / `Tensor<D, Int>`                                                    |
| `Model<B>`, `Linear<B>`, `Module<B>`                | `Model`, `Linear`, `Module`                                                       |
| `B::Device`                                         | `burn::tensor::Device`                                                            |
| A backend type alias at the application entry point | A constructor such as `Device::wgpu(...)`, `Device::cuda(0)`, or `Device::flex()` |
| `Autodiff<B>` as the application's backend          | `device.autodiff()` before model/input initialization                             |
| Backend element type parameters                     | Device dtype defaults, explicit creation dtypes, and `tensor.cast(...)`           |
| `tensor.into_scalar()`                              | `tensor.into_scalar::<f32>()`, or an inferred element type                        |
| Backend-generic optimizers                          | `ModuleOptimizer`, returned by optimizer configuration `init()`                   |
| `burn::module::Initializer`                         | `burn::nn::Initializer`                                                           |

Device-level operations previously called through `B: Backend`, such as seeding and synchronization,
are now methods on `Device`. See [Using a Device](./building-blocks/backend.md#using-a-device).

When upgrading a model, remove its backend parameter and the corresponding parameters on fields and
methods. The rank and kind remain part of the tensor type:

```rust,ignore
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
struct Model {
    linear: Linear,
}

impl Model {
    fn new(device: &Device) -> Self {
        Self { linear: LinearConfig::new(4, 2).init(device) }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.linear.forward(input)
    }
}

// Requires the flex and autodiff features.
let device = Device::flex().autodiff();
let model = Model::new(&device);
let input = Tensor::<2>::ones([8, 4], &device);
let gradients = model.forward(input).sum().backward();
assert!(model.linear.weight.grad(&gradients).is_some());
```

Configure device dtype defaults before creating tensors. Configuration is shared by the compute
device and can only be initialized once. See [Backend and Device](./building-blocks/backend.md).

Prefer explicit device constructors during migration. `Device::default()` chooses from compiled-in
backends, not from available hardware. Enabling an additional backend through Cargo feature
unification can therefore change the default. This also affects implicit device selection by
`Tensor::from(...)` and dataloaders without `set_device(...)`.

## Autodiff is runtime state

Removing `B: AutodiffBackend` moves precondition checks to runtime. Enabling autodiff permits graph
recording; it does not make every tensor require gradients.

| Question                                    | API                                  |
| ------------------------------------------- | ------------------------------------ |
| Does this tensor have autodiff enabled?     | `is_autodiff()`                      |
| Does it participate in a graph?             | `is_tracked()` on float tensors      |
| Will its gradient be retained?              | `is_require_grad()` on float tensors |
| Which checkpointing strategy does it carry? | `gradient_checkpointing_strategy()`  |

Call `require_grad()` on source leaves **before** computing the output. Calling it on a plain float
tensor or tracked intermediate panics. Derived tensors can be tracked without retaining their own
gradient. `set_require_grad(false)` cuts the graph, while `detach()` cuts the graph and preserves a
leaf's retention setting. Use `without_autodiff()` to remove the association entirely; `inner()`
remains an alias. `autodiff()` is the clearer spelling of `from_inner(...)`.

`backward()` consumes reachable graph steps even though it borrows the output. Clones share the
tape; `is_tracked()` remains true after consumption. Recompute a forward pass for another backward,
or combine losses sharing intermediates before backward. See
[Autodiff](./building-blocks/autodiff.md).

## Moving tensors and switching module state

`to_device()` preserves a tensor's source autodiff and checkpointing context. Moving a plain tensor
to an autodiff device does not enable autodiff; moving an autodiff tensor to a plain device does not
disable it. Device equality ignores those settings.

For initialized module parameters, `to_device()` preserves gradient connections to source
parameters. Its tracked output parameters are intermediates that cannot themselves be optimized. An
uninitialized, unshared parameter instead initializes directly on the destination and remains a
leaf. Cloning a lazy parameter shares its initialization state; moving a shared parameter
initializes it on the source first. Use `fork(&destination)` for independent destination leaves
regardless of initialization state. Starting from a plain or validation module, use
`model.train().fork(&destination)` to enable training explicitly.

`AutodiffModule` has been merged into `Module`. Replace its imports and bounds with `Module`;
replace module `from_inner(module)` calls with `module.train()`. Tensor `from_inner` remains
available. A `Module` bound does not establish that a value is currently training. `valid()` and
`train()` return the same type. `valid()` disables autodiff and training flags in a snapshot;
`train()` restores configured trainability and flags. Explicit `no_grad()` and `freeze()` settings
persist. `freeze()` also disables module-owned training flags, whereas `no_grad()` only changes
parameter gradients.

Keep the original training model when using `model.valid()` for validation. The snapshot discards
tensor checkpointing strategies, which `train()` does not restore.

Dropout additionally checks its input tensor's autodiff context, so create model inputs on the
training device even when their gradients are not needed. See [Module](./building-blocks/module.md).

## Migrating checkpoints

Records now use burnpack and `ModuleRecord`; the old `Recorder`, `PrecisionSettings`, generated
record types, and `#[derive(Record)]` are removed. Burn 0.22 cannot directly load their MessagePack
(`.mpk`), binary (`.bin`), or JSON files, including compressed variants. Renaming a file to `.bpk`
does not convert it.

For burnpack checkpoints, loading and saving no longer require a recorder. Weights are loaded onto
the model's devices.

| Previous API                                 | 0.22 API                     |
| -------------------------------------------- | ---------------------------- |
| `model.load_file(path, &recorder, &device)?` | `model.try_load_file(path)?` |
| `model.save_file(path, &recorder)?`          | `model.save_file(path)?`     |

Use `try_load_file` to handle loading errors; `load_file` now panics on failure.

Loaded weights retain the checkpoint's dtype by default. To use the model's dtype instead, load a
`ModuleRecord` and call `cast_to_module_dtype()` before applying it. See
[Saving and Loading Models](./saving-and-loading.md).

To migrate model weights, load the checkpoint in an older Burn project that can read it, using the
original model definition, recorder, and precision settings. Export the loaded model through
`burn-store`, then import those weights into the corresponding 0.22 model.

For example, for a checkpoint readable by Burn 0.21, add `burn-store = "0.21"` to that project's
dependencies. After initializing the original model and device, load and export its parameters:

```rust,ignore
// Run in the Burn 0.21 project.
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_store::{ModuleSnapshot, SafetensorsStore};

// Match the recorder and precision settings used to save the checkpoint.
// This example reads model.mpk; load_file adds the recorder's extension.
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
let model = model.load_file("model", &recorder, &device)?;

let mut store = SafetensorsStore::from_file("model.safetensors");
model.save_into(&mut store)?;
```

If the checkpoint was saved with `CompactRecorder`, use `HalfPrecisionSettings` instead. For other
formats, use the corresponding recorder. Checkpoints from earlier releases may require an
intermediate migration before 0.21 can read them.

In the Burn 0.22 project, add `burn-store = "0.22"`, initialize the migrated model, and load the
exported weights:

```rust,ignore
// Run in the Burn 0.22 project, with an initialized mutable model.
use burn_store::{ModuleSnapshot, SafetensorsStore};

let mut store = SafetensorsStore::from_file("model.safetensors");
model.load_from(&mut store)?;
```

Burn-produced SafeTensors files need no PyTorch adapter. Keep parameter paths and shapes consistent
between the models, check the loaded dtypes, and compare outputs on the same input after conversion.
See [Saving and Loading Models](./saving-and-loading.md) for key remapping and other loading
options.

If you already exported weights through `burn-store`, use `SafetensorsStore` or `BurnpackStore` for
the corresponding format. Burnpack exports from 0.21 use compatible tensor metadata, but this does
not guarantee compatibility with every earlier `.bpk` file: older releases may encode dtypes
differently.

This procedure transfers model parameters. It does not migrate legacy optimizer or learning rate
scheduler records, so it does not resume the full training checkpoint. Start with new optimizer and
scheduler state, or implement a separate conversion if preserving that state is required. See
[Record](./building-blocks/record.md) for the 0.22 record APIs.

## Datasets and dataloaders

Dataset access is now fallible. Update custom datasets and training loops to handle these return
types:

| API                      | Previous    | 0.22                          |
| ------------------------ | ----------- | ----------------------------- |
| `Dataset::get`           | `Option<I>` | `Result<I, E>`                |
| Dataset iterator item    | `I`         | `Result<I, E>`                |
| Dataloader iterator item | `Batch`     | `Result<Batch, DatasetError>` |

In custom datasets, return `Ok(item)` on success and `Err(error)` for retrieval failures.
Out-of-bounds access must panic. The error type `E` defaults to `DatasetError`.

In custom training loops, handle each batch's result before using it, for example with
`let batch = batch?;`. See [Dataset](./building-blocks/dataset.md) for examples and dataloader
compatibility with custom error types.

For specialized dataset APIs:

- **Windows:** `window()` returns `Result<Option<Vec<I>>, DatasetError>`. Handle the error before
  checking whether a window fits. Window iterator items are also wrapped in `Result`.
- **Images:** replace `Vec<PixelDepth>` with `PixelData` when constructing `ImageDatasetItem`. Match
  its `U8`, `U16`, or `F32` variant for the packed values, or use `image.iter()` for individual
  `PixelDepth` values. See [Images](./building-blocks/dataset.md#images).
- **SQLite:** enable `sqlite` explicitly. Update error matches: `SqliteDatasetError::Sql` wraps
  `turso::Error`, and `Row` and `Deserialize` are new variants.

## Training

Update your training configuration:

| Previous API                                     | 0.22 API                                                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `with_file_checkpointer(CompactRecorder::new())` | `with_default_checkpointers()`                                                                         |
| `renderer(renderer)`                             | `renderer(Box::new(renderer))`                                                                         |
| `AurocMetric::new()`                             | `AurocMetric::binary()`, `AurocMetric::multiclass(reduction)`, or `AurocMetric::multilabel(reduction)` |
| `AurocInput`                                     | `ClassificationOutput` or `MultiLabelClassificationOutput`                                             |

Default checkpointers save the model, optimizer, and scheduler as burnpack files. AUROC's multiclass
and multilabel constructors take a `ClassReduction`. See [Learner](./building-blocks/learner.md) for
training configuration.

Review configurations and numerical baselines affected by these behavior changes:

- **Cosine annealing:** cycles no longer restart abruptly. `num_iters` specifies the descent to the
  minimum, after which the rate rises along the cosine curve. Update schedules that relied on warm
  restarts. See [Cosine annealing](./building-blocks/lr-scheduler.md#cosine-annealing).
- **Padding:** cross-entropy loss and accuracy exclude padded targets from their denominators,
  including the weight sum for weighted loss. An entirely padded cross-entropy batch returns NaN.

## Tensor data and numeric semantics

Replace the deprecated `TensorData` vector methods and handle their errors:

| Previous API           | 0.22 API                   |
| ---------------------- | -------------------------- |
| `data.to_vec::<E>()`   | `data.try_to_vec::<E>()`   |
| `data.into_vec::<E>()` | `data.try_into_vec::<E>()` |

These methods return `Result<Vec<E>, DataError>` and require `E` to match the stored dtype. For
conversion, use `try_to_vec_as::<E>()` or `try_into_vec_as::<E>()` on `TensorData` or `Tensor`.
Update error matches for the revised `DataError` variants and `Tensor::try_into_scalar`'s
`TensorReadError`.

`TensorData` fields are private, so its byte length always matches its shape and dtype (quantized
data is not checked yet). Replace field access with the accessors:

| Previous API                         | 0.22 API                                                  |
| ------------------------------------ | --------------------------------------------------------- |
| `data.shape`                         | `data.shape()` (returns `&Shape`)                         |
| `data.dtype`                         | `data.dtype()`                                            |
| `data.bytes` (borrowed)              | `data.bytes()` or `data.as_bytes()`                       |
| `data.bytes` (moved)                 | `data.into_bytes()`, or `data.into_parts()` for all three |
| `&mut data.bytes`                    | `TensorData::with_bytes_mut(..)` (length must not change) |
| `TensorData { bytes, shape, dtype }` | `TensorData::try_from_bytes(bytes, shape, dtype)?`        |

`TensorData::from_bytes` and `from_bytes_vec` now panic when the byte length does not match the
shape and dtype. Use `try_from_bytes` or `try_from_bytes_vec` for untrusted input; they return
`DataError::InvalidByteLength`, the same check deserialization applies.

Other source changes:

- **Dimensions:** negative indices are supported. Annotate untyped empty inputs, such as
  `flip([] as [isize; 0])` or `squeeze_dims(&[] as &[isize])`.
- **Convolution:** `ConvOptions::padding` stores `(before, after)` pairs. Keep
  `ConvOptions::new(..)` for symmetric padding; replace deprecated `PaddedConvOptions` with
  `ConvOptions::new_with_padding(..)` for asymmetric padding.
- **Quantization:** replace `with_level(..)` and `with_param(..)` with `per_tensor(ScaleDtype)` or
  `per_block(block, ScaleDtype)`. See [Quantization](./performance/quantization.md).
- **Softplus:** use `SoftplusConfig::new().with_beta(beta).with_threshold(threshold)` instead of
  struct literals to account for the new `threshold` field (default: 20).

Update numerical expectations for these cases:

| Case                                          | 0.22 behavior                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------- |
| NaN in a `max`, `min`, or `max_abs` reduction | Returns NaN                                                                           |
| NaN in an `argmax` or `argmin` reduction      | Returns the first NaN's index                                                         |
| NaN in `cummax` or `cummin`                   | Returns NaN from that position onward                                                 |
| Reducing a zero-length axis                   | `sum`: 0; `prod`: 1; `any`: false; `all`: true; float `mean`: NaN; `max`/`min`: panic |
| Empty axes in `max_abs_dims` or `*_norm_dims` | Applies the elementwise transformation without reducing                               |

## Custom integrations

The following sections apply when you implement Burn traits yourself or use lower-level APIs. Skip
them if your project only uses the built-in modules, optimizers, metrics, and stores.

### Modules

For handwritten implementations, consult the `Module` trait documentation for the required methods;
`#[derive(Module)]` generates them automatically.

Other module API changes:

- `BatchNorm` and `Dropout` now include `Param<Flag>` training controls; use their config builders
  instead of struct literals.
- Replace `ParamId::serialize()` / `deserialize()` with `Display` / `FromStr`.
- Replace `Reinitializer` with `burn::nn::Initializer` for new parameters or a `ModuleMapper` for
  existing ones. Use `Param::map` to preserve IDs and configured trainability, and keep trainable
  tensors as gradient-retaining leaves. See the
  [mapper example](./building-blocks/module.md#visitor--mapper).

`Parameter` is sealed through `ParameterValue`. Replace custom `Param<T>` types with tensor
parameters, `Param<Flag>` for training controls, or ordinary module fields for other state.

### Optimizers and schedulers

Replace `SimpleOptimizer` / `OptimizerAdaptor` with the per-tensor `Optimizer` trait and its
`ModuleOptimizer` wrapper. Use `RecordState` and the new record types instead of `#[derive(Record)]`
for optimizer and scheduler state. See [Optimizer](./building-blocks/optimizer.md).

For custom `LrScheduler` implementations, remove the associated `Record` type and backend-generic
record methods. Implement:

- `to_record(&self) -> LrSchedulerRecord`
- `load_record(&mut self, record: LrSchedulerRecord)` — loading now mutates the scheduler.

Derive `Clone` to obtain the blanket `LrSchedulerClone` implementation.

### Custom metrics

Update the metric lifecycle:

- Implement `Metric::compute(&mut self)` to compute the epoch value after per-batch updates.
- Return `Option<NumericEntry>` from `Numeric::value()` and `running_value()`. Use `None` when the
  metric is only defined at the end of an epoch.
- Return the computed epoch value from `final_value()`.

See [Custom Metric](./building-blocks/metric.md#custom-metric) for an implementation example.

### Renderers and event processors

Implement `TrainingProgressLogger` and `EvaluationProgressLogger` for custom `MetricsRenderer`
types. Move progress handling from the removed `render_train`, `render_valid`, and `render_test`
methods to logger callbacks. The old `TrainingProgress`, `EvaluationProgress`, and `ProgressType`
types are removed.

Update custom event matches:

| Event                                   | Required change                                         |
| --------------------------------------- | ------------------------------------------------------- |
| `LearnerEvent::Start`                   | Match struct fields `total_epochs` and `starting_epoch` |
| `EvaluatorEvent::Start`                 | Match struct field `total_tests`                        |
| `LearnerEvent::StartSplit` / `EndSplit` | Handle the new split lifecycle events                   |
| `EvaluatorEvent::StartTest` / `EndTest` | Handle the new test lifecycle events                    |

### Distributed training

Remove the `distributed` and `collective` feature flags and the `burn-collective` dependency.
Collective operations are available through `burn::tensor::distributed`.

- Replace `DistributedSession` / `DistributedRuntime` with `DistributedContext`.
- Construct DDP strategies with `ExecutionStrategy::ddp(devices, DistributedConfig { .. })`.
- Call `.into_vec()` on `Device::enumerate(..)` when a `Vec<Device>` is required.

Check runtime support before using collectives: CubeCL all-reduce currently requires CUDA, including
on remote servers. See [Distributed Computing](./performance/distributed-computing.md).

### Storage adapters and checkpointers

Update custom `burn-store` integrations:

| Previous API        | 0.22 API                                                |
| ------------------- | ------------------------------------------------------- |
| `TensorSnapshot`    | `burn_pack::Tensor`, including in `collect` and `apply` |
| `get_snapshot`      | `get_tensor`                                            |
| `get_all_snapshots` | `get_all_tensors`                                       |

`ModuleAdapter::adapt` now takes a `burn_pack::Tensor` and `ModuleContext<'_>`, and returns the
tensor. Remove the `burnpack` feature flag; support is always enabled. PyTorch readers now use the
`pytorch-reader` crate. See [Saving and Loading Models](./saving-and-loading.md).

For custom training checkpoints, pass implementations of these traits to
`SupervisedTraining::with_custom_checkpointers(..)`:

- `Checkpointer<ModuleRecord>`
- `Checkpointer<OptimizerRecord>`
- `Checkpointer<LrSchedulerRecord>`

### Backend extensions

To migrate custom operations:

1. Enable `extension` for low-level access through `burn::backend`.
2. Annotate your backend trait with `#[backend_extension(...)]` and implement it on the supported
   backends. Use the `Cube` selector for CubeCL runtimes such as WGPU and CUDA.
3. Expose a `Tensor<D>` wrapper using `Dispatch` and `into_dispatch()` / `from_dispatch()`.

The macro generates routing. Custom derivatives still need an autodiff implementation. See
[Backend Extension](./advanced/backend-extension/) for a complete example and Fusion requirements.

Update low-level tensor access:

- Specify the backend when constructing a tensor: `Tensor::from_primitive::<B>(primitive)`.
- Replace `into_primitive` with `try_into_primitive::<B>()` and handle backend mismatches. Shared
  CubeCL backend aliases do not establish which runtime the tensor uses.
- Remove backend parameters from `TensorKind` bounds. The trait is sealed and its `Primitive`
  associated type is removed; use the conversion methods above for primitives and `KIND` to inspect
  the tensor kind.
- Use accessors for the now-private `AutodiffTensor` fields.

Out-of-tree backends are not supported in Burn 0.22. `#[backend_extension]` adds operations to
Burn's supported backends; it does not currently allow registering new backends.
