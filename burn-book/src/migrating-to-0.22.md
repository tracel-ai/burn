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

Scalar readback is now generic over the returned element type. Specify it explicitly with
`tensor.into_scalar::<f32>()` or let a typed binding infer it:
`let value: f32 = tensor.into_scalar();`. The same applies to asynchronous and fallible scalar
readback methods.

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

For modules, `to_device()` preserves gradient connections to source parameters. Its tracked output
parameters are intermediates that cannot themselves be optimized. Use `fork(&destination)` for
independent destination leaves. Starting from a plain or validation module, use
`model.train().fork(&destination)` to enable training explicitly.

`AutodiffModule` has been merged into `Module`. Replace its imports and bounds with `Module`;
replace module `from_inner(module)` calls with `module.train()`. Tensor `from_inner` remains
available. A `Module` bound does not establish that a value is currently training. `valid()` and
`train()` return the same type. `valid()` disables autodiff and training flags in a snapshot;
`train()` restores configured trainability and flags. Explicit `no_grad()` and `freeze()` settings
persist. `freeze()` also disables module-owned training flags, whereas `no_grad()` only changes
parameter gradients.

Keep the original training model when using `model.valid()` for validation. The snapshot folds
adapters such as LoRA into parameter values and discards checkpointing strategies;
`snapshot.train()` does not reconstruct those. Dropout additionally checks its input tensor's
autodiff context, so create model inputs on the training device even when their gradients are not
needed. See [Module](./building-blocks/module.md).

## Datasets and dataloaders

`Dataset<I>` now has an optional error type parameter: `Dataset<I, E = DatasetError>`. Change
`get(&self, index: usize) -> Option<I>` implementations to return `Result<I, E>`. Return `Ok(item)`
for an in-bounds item and `Err(error)` for a retrieval failure, such as an I/O or decoding error.
Accessing `index >= len()` must panic; it no longer returns `None`. A custom error type must
implement `std::error::Error + Send + Sync + 'static`.

Dataset iterators yield `Result<I, E>`, and dataloader iterators yield
`Result<Batch, DatasetError>`. In custom loops, handle each result before passing the item or batch
to your model, for example with `let batch = batch?;` in a function returning `Result`. An iterator
returns `None` only when it is exhausted; a retrieval error is an item to handle, not an end-of-data
marker. See [Dataset](./building-blocks/dataset.md) for the updated trait and a fallible batch loop.

`ImageDatasetItem.image` now stores `PixelData` instead of `Vec<PixelDepth>`. Match `PixelData::U8`,
`PixelData::U16`, or `PixelData::F32` to access the packed vector, and construct the matching
variant when producing image items. For depth-independent processing, `image.iter()` still yields
individual `PixelDepth` values. See [Images](./building-blocks/dataset.md#images).

`SqliteDataset` is backed by Turso instead of `rusqlite`, so `SqliteDatasetError::Sql` now wraps
`turso::Error`, and the `Row` and `Deserialize` variants are new. Its `sqlite` feature is no longer
enabled by `dataset`; see [Cargo features and toolchain](#cargo-features-and-toolchain).

## Migrating checkpoints

Records now use burnpack and `ModuleRecord`; the old `Recorder`, `PrecisionSettings`, generated
record types, and `#[derive(Record)]` are removed. Burn 0.22 cannot directly load their MessagePack
(`.mpk`), binary (`.bin`), or JSON files, including compressed variants. Renaming a file to `.bpk`
does not convert it.

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

## Training with `burn-train`

`SupervisedTraining` no longer takes a recorder for checkpointing. Replace
`with_file_checkpointer(CompactRecorder::new())` with `with_default_checkpointers()`, which writes
burnpack files for the model, optimizer, and scheduler under the artifact directory. `renderer(..)`
now takes a `Box<dyn MetricsRenderer>`, and `with_progress_logger(..)` registers a
`TrainingProgressLogger` that observes the training lifecycle. See
[Learner](./building-blocks/learner.md).

`AurocMetric::new()` and `AurocInput` are removed. Construct the metric with
`AurocMetric::binary()`, `multiclass(ClassReduction)`, or `multilabel(ClassReduction)`; it adapts
from `ClassificationOutput` and `MultiLabelClassificationOutput` like the other classification
metrics.

`CosineAnnealingLrScheduler` no longer resets to the initial learning rate after each cycle.
`num_iters` is the number of steps from the initial rate to the minimum; subsequent steps continue
along the cosine curve and the rate rises again. Existing 0.21 configurations that relied on warm
restarts will produce a different schedule. See
[Cosine annealing](./building-blocks/lr-scheduler.md#cosine-annealing).

## Tensor data and numeric semantics

`TensorData::to_vec` and `into_vec` are deprecated in favor of `try_to_vec` and `try_into_vec`,
which return `Result<Vec<E>, DataError>` and fail when `E` does not match the stored dtype. Use
`try_to_vec_as::<E>()` and `try_into_vec_as::<E>()` to convert to another element type, on both
`TensorData` and `Tensor`. `Tensor::try_into_scalar` now returns `TensorReadError`, and the
`DataError` variants were reworked.

Extrema reductions propagate NaN on every backend: `max`, `min`, and `max_abs` return NaN when the
reduced slice contains one, `argmax` and `argmin` return the index of the first NaN, and `cummax`
and `cummin` are NaN from the first NaN onward. Reducing a zero-length axis returns the identity for
`sum` (0), `prod` (1), `any` (false), and `all` (true), NaN for a float `mean`, and panics for `max`
and `min`. `max_abs_dims(&[])` and the `*_norm_dims(&[])` variants apply the elementwise
transformation without reducing.

Dimension arguments accept negative indices across the tensor API, counting from the last axis. Most
calls are source-compatible; untyped empty inputs now need an annotation, for example
`flip([] as [isize; 0])` or `squeeze_dims(&[] as &[isize])`.

`ConvOptions::padding` is now `[(usize, usize); N]`, holding the padding at the beginning and end of
each spatial dimension. `ConvOptions::new(..)` still takes symmetric padding; use
`ConvOptions::new_with_padding(..)` for asymmetric padding. `PaddedConvOptions` is deprecated.

Quantization schemes select their granularity and scale dtype with `per_tensor(ScaleDtype)` and
`per_block(block, ScaleDtype)` instead of `with_level(..)` and `with_param(..)`. See
[Quantization](./performance/quantization.md).

## Custom integrations

The following sections apply when you implement Burn traits yourself or use lower-level APIs. Skip
them if your project only uses the built-in modules, optimizers, metrics, and stores.

### Modules and optimizers

Handwritten `Module` implementations now implement `valid(&self)` and `train(self)`; the derive
generates both. `ParamId::serialize()` and `deserialize()` are replaced by its `Display` and
`FromStr` implementations. Optimizer implementations use the per-tensor `Optimizer` trait, wrapped
by `ModuleOptimizer`. See [Module](./building-blocks/module.md) and
[Optimizer](./building-blocks/optimizer.md).

### Custom metrics

Custom metrics must implement `Metric::compute(&mut self)`, which produces the epoch value after the
per-batch `update` calls. `Numeric::value()` and `running_value()` now return
`Option<NumericEntry>`, returning `None` for metrics that are only defined at the end of an epoch,
and `final_value()` returns the computed epoch value. See
[Custom Metric](./building-blocks/metric.md#custom-metric).

### Renderers and event processors

Custom `MetricsRenderer` implementations must also implement `TrainingProgressLogger` and
`EvaluationProgressLogger`. The `render_train`, `render_valid`, and `render_test` methods and the
`TrainingProgress`, `EvaluationProgress`, and `ProgressType` types are removed; progress arrives
through the logger callbacks instead. In custom event processors, `LearnerEvent::Start` and
`EvaluatorEvent::Start` are now struct variants carrying `total_epochs`/`starting_epoch` and
`total_tests`, and the `LearnerEvent::StartSplit`, `LearnerEvent::EndSplit`,
`EvaluatorEvent::StartTest`, and `EvaluatorEvent::EndTest` variants are new.

### Distributed training

The `distributed` and `collective` Cargo features and the `burn-collective` crate are gone;
collective operations live in `burn::tensor::distributed` without a feature flag.
`DistributedSession` and `DistributedRuntime` are replaced by `DistributedContext`, and the DDP
strategy is created with `ExecutionStrategy::ddp(devices, DistributedConfig { .. })`.
`Device::enumerate(..)` returns a `Devices` wrapper; call `into_vec()` to obtain the `Vec<Device>`.
See [Distributed Computing](./performance/distributed-computing.md).

### Storage adapters and checkpointers

`burn-store` transports tensors as `burn_pack::Tensor` instead of `TensorSnapshot`: `collect` and
`apply` take and return it, `get_snapshot` and `get_all_snapshots` are renamed to `get_tensor` and
`get_all_tensors`, and `ModuleAdapter::adapt` receives the tensor and a borrowed `ModuleContext`.
The `burnpack` feature is gone; burnpack support is always available. PyTorch checkpoints are read
through the `pytorch-reader` crate. See [Saving and Loading Models](./saving-and-loading.md).

To store training checkpoints elsewhere than the default burnpack files, pass
`Checkpointer<ModuleRecord>`, `Checkpointer<OptimizerRecord>`, and `Checkpointer<LrSchedulerRecord>`
implementations to `SupervisedTraining::with_custom_checkpointers(..)`.

### Backend extensions

Enable the `extension` feature. Use `#[backend_extension(...)]` on your low-level trait, implement
it on the supported backends, and expose a `Tensor<D>` wrapper using `Dispatch` and
`into_dispatch()` / `from_dispatch()`. The macro generates routing; it does not generate a custom
derivative. The CubeCL selector is `Cube`, covering runtime aliases such as WGPU and CUDA. See
[Backend Extension](./advanced/backend-extension/) for a complete example and Fusion requirements.

`Tensor::from_primitive` is generic over the backend, for example
`Tensor::from_primitive::<B>(primitive)`. `into_primitive` is replaced by
`try_into_primitive::<B>()`, which returns an error when the tensor is not on backend `B`.
`TensorKind` no longer has a backend type parameter or a `Primitive` associated type, is sealed, and
identifies the kind through the `TensorKind::KIND` constant of type `Kind`. `AutodiffTensor` fields
are no longer public.
