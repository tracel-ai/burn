# Changelog

Changes for the next release. For published releases, see
[GitHub Releases](https://github.com/tracel-ai/burn/releases).

## Unreleased

### Interpolation options

**Breaking:** `InterpolateMode` variants now own their coordinate transform and cubic coefficient
`a`. `Nearest` and `NearestExact` have fixed mappings and no overrides.
Configurable coordinate transforms are limited to `HalfPixel` and `AlignCorners`;
asymmetric coordinates remain the fixed mapping of `Nearest`.

```diff
+use burn::tensor::ops::CoordinateTransformMode;
-InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false)
+InterpolateOptions::new(InterpolateMode::Bilinear {
+    coordinate_transform: CoordinateTransformMode::HalfPixel,
+})
```

Use `AlignCorners` for the previous `true`/default alignment and `a: -0.75` for the previous cubic
coefficient. For `Interpolate1dConfig` and `Interpolate2dConfig`, move these settings into
`InterpolateMode::{Linear, Cubic, Lanczos}`; update serialized configurations accordingly.
Grid sampling now uses `GridSampleMode::{Nearest, Bilinear}`.

Explicit scales now also drive scale-dependent coordinate mapping. Use `.with_output_size(...)`
to retain size-derived mapping. See the [interpolation guide](building-blocks/module.md#interpolation)
for configuration and backend limitations.
([#6154](https://github.com/tracel-ai/burn/pull/6154), [@Sadik00789](https://github.com/Sadik00789))

### Device-time instrumentation

`burn-instrument` and `#[compute]` annotations measure device time by model component. Measurement
files and reports can be analyzed without a runtime. Enable Burn's `instrument` feature to profile
built-in layers without adding annotations; annotations compile away when the feature is disabled.
Reports also include the measured cost of a kernel launch.
([#6131](https://github.com/tracel-ai/burn/pull/6131),
[#6143](https://github.com/tracel-ai/burn/pull/6143),
[#6165](https://github.com/tracel-ai/burn/pull/6165),
[#6201](https://github.com/tracel-ai/burn/pull/6201))
[@nathanielsimard](https://github.com/nathanielsimard) [@louisfd](https://github.com/louisfd)

- Improve CUDA kernel timing by excluding host resource preparation, and correct GPU bandwidth
  probes that could overestimate throughput by repeatedly reading cached data.
  ([#6193](https://github.com/tracel-ai/burn/pull/6193),
  [CubeCL #1914](https://github.com/tracel-ai/cubecl/pull/1914),
  [CubeCL #1915](https://github.com/tracel-ai/cubecl/pull/1915))
  [@louisfd](https://github.com/louisfd)
- Reports no longer score a component that is only partly fused, such as a layer projection that
  ran alone in one layer and fused with the residual add in the others; its row reads
  `partly fused ↓` instead of a percentage far past the peak. `FusedElsewhere::contains` is replaced
  by `share`, which returns a `FusedShare`. ([#6222](https://github.com/tracel-ai/burn/pull/6222))
  [@nathanielsimard](https://github.com/nathanielsimard)
- `Report::components` and `Report::unclaimed` return each component as the report's table reads
  it — its path, its time, its share of the work (a `WorkShare`) and of the host, how much of it
  fused elsewhere, its score and floor — and each kernel under the name the table prints, so a
  front door can lay a report out its own way. A component whose work all fused elsewhere reads
  no time or share, as the table prints `fused ↓`. The table is printed from the same readings.
  ([#6260](https://github.com/tracel-ai/burn/pull/6260))
  [@nathanielsimard](https://github.com/nathanielsimard)

### Execution performance

#### Grouped-query attention and causal masks

Grouped-query attention now works across backends and autodiff, with native CubeCL acceleration.
Query heads can share fewer key/value heads; invalid head counts are rejected.

**Breaking:** `AttentionModuleOptions` gains `causal_alignment`. Add
`causal_alignment: CausalAlignment::BottomRight` to struct literals to preserve existing behavior,
or use `..Default::default()`. Select `TopLeft` for causal masks aligned to the first query/key
positions when sequence lengths differ.
([#6127](https://github.com/tracel-ai/burn/pull/6127),
[#5953](https://github.com/tracel-ai/burn/pull/5953))
[@gzsombor](https://github.com/gzsombor) [@Andy2887](https://github.com/Andy2887)
[@louisfd](https://github.com/louisfd)

#### Kernels

- Avoid a full-sized gradient copy in CubeCL FP32 `conv2d` bias backward, while retaining
  single-pass accumulation for FP16/BF16 to avoid intermediate overflow.
  ([#6173](https://github.com/tracel-ai/burn/pull/6173))
  [@EthanAlmloff](https://github.com/EthanAlmloff)
- Enable CUDA tensor-core acceleration on Blackwell and Blackwell Ultra GPUs, including B300.
  ([#6161](https://github.com/tracel-ai/burn/pull/6161))
- Improve CubeCL CPU matrix multiplication by selecting tile sizes that fit the vector width
  and using GEMM settings tailored to CPU hardware.
  ([#6177](https://github.com/tracel-ai/burn/pull/6177),
  [#6245](https://github.com/tracel-ai/burn/pull/6245))
  [@Charles23R](https://github.com/Charles23R)
- Speed up CubeCL `conv_transpose3d` and reduce Flex `conv2d` memory use and layout conversions.
  ([#5793](https://github.com/tracel-ai/burn/pull/5793),
  [#6075](https://github.com/tracel-ai/burn/pull/6075))
- Run GELU and sigmoid forward and backward operations in single CubeCL kernels.
  ([#6114](https://github.com/tracel-ai/burn/pull/6114)) [@ssoj13](https://github.com/ssoj13)
- Allow more CubeCL slices to share their source buffer, avoiding copies of KV-cache prefixes.
  ([#6136](https://github.com/tracel-ai/burn/pull/6136))
  [@ssoj13](https://github.com/ssoj13) [@laggui](https://github.com/laggui)
- Reduce fusion's CPU overhead by skipping unnecessary reduction-fuser resets.
  ([#6036](https://github.com/tracel-ai/burn/pull/6036)) [@louisfd](https://github.com/louisfd)
- Add the experimental `burn-cubecl/tile-kernels-only` feature for benchmarking CubeK's new matmul
  and attention routines. It is disabled by default and rejects unsupported inputs without a
  fallback; fused matmul and convolution are unchanged. Quantized matmul inputs are read packed
  where the routines can, and dequantized first elsewhere. Attention takes a bool mask and both
  causal alignments.
  ([#6193](https://github.com/tracel-ai/burn/pull/6193),
  [#6199](https://github.com/tracel-ai/burn/pull/6199),
  [#6246](https://github.com/tracel-ai/burn/pull/6246)) [@louisfd](https://github.com/louisfd)
- Run attention's backward pass on CubeCL backends as CubeK kernels from the forward's saved
  logsumexp, rather than autodiff over the decomposed attention. The kernels take f32, f16 and
  bf16 attention without a mask, bias or softcap, causal or not; a softcap, a top-left causal
  mask over sequences of different lengths or an element the kernels do not take is composed from
  tensor operations, and a mask or an additive bias keeps the decomposed gradients. Autotune
  may pick a routine that sums the query gradient with atomic adds, so its last bits can differ
  from run to run. Backends can provide their own through `ModuleOps::attention_with_logsumexp`
  and `ModuleOps::attention_backward`. [@louisfd](https://github.com/louisfd)

#### Compilation and autotuning

- Compile autotuning candidates in parallel, including CUDA binary assembly, to reduce startup time.
  ([#5996](https://github.com/tracel-ai/burn/pull/5996),
  [#6074](https://github.com/tracel-ai/burn/pull/6074))
- Reuse compiled CPU kernels across process runs when persistent compilation caching is enabled.
  ([#6099](https://github.com/tracel-ai/burn/pull/6099))
- Share autotuning results across devices with the same hardware. Previously saved per-device
  results are tuned again once. ([#6150](https://github.com/tracel-ai/burn/pull/6150))
- Avoid excessive kernel compilation for large depthwise-convolution filters.
  ([#6072](https://github.com/tracel-ai/burn/pull/6072))
- Stop dumping LLVM compiler IR into normal experiment logs, reducing compilation overhead and log
  size. ([#6188](https://github.com/tracel-ai/burn/pull/6188),
  [CubeCL #1895](https://github.com/tracel-ai/cubecl/pull/1895))

### Runtime configuration

Matrix multiplication precision is now configurable through `[cubek.matmul].allow_tf32` in
`burn.toml`. Set it to `false` to disable automatic F32-to-TF32 conversion; the default is `true`.
([#6170](https://github.com/tracel-ai/burn/pull/6170))

```toml
# burn.toml
[cubek.matmul]
allow_tf32 = false
```

See [runtime configuration](./advanced/runtime-configuration.md#cubek-matmul-precision) for details.

### Distributed and remote execution

#### Layer parallelism

Models can now be split across devices by layer, enabling training and inference with models that
exceed a single device's memory. The new `LayerParallelism` trait and `DistributedLayeredModel`
wrapper support sequential execution, transferring activations between devices.
`LayerPipeline` streams several inference batches through such a model with a thread per stage,
so the stages overlap. Training schedules and `Learner` integration are not included.
([#5702](https://github.com/tracel-ai/burn/pull/5702),
[#6277](https://github.com/tracel-ai/burn/pull/6277))
[@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Distributed collectives

**Breaking:** `distributed::all_reduce` and `CollectiveTensor` have been removed. Use
`Tensor::all_reduce(tensors, op)`, passing all participating tensors together. Only one
`DistributedContext` may be alive per backend in a process.
([#6144](https://github.com/tracel-ai/burn/pull/6144),
[#6183](https://github.com/tracel-ai/burn/pull/6183))

`Tensor::all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all`, and `broadcast` provide
collectives across a group of devices. Reductions support float and integer tensors; gather,
all-to-all, and broadcast also support boolean tensors. Quantized tensors are rejected.
([#6183](https://github.com/tracel-ai/burn/pull/6183))

- Fix DDP gradient-synchronization hangs across local and remote devices.
  ([#6144](https://github.com/tracel-ai/burn/pull/6144))
- Coordinate DDP batch availability before backward passes and synchronize epoch transitions.
  Workers must yield the same number of training batches; uneven exhaustion now reports an error.
  ([#6184](https://github.com/tracel-ai/burn/pull/6184))
- Drain every device's pending fusion work at once before a collective, speeding up
  tensor-parallel decode. ([#6293](https://github.com/tracel-ai/burn/pull/6293))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Device groups

- A tensor made from data on a device group stays on the host until it is used, so `from_data`
  followed by `with_sharding` uploads each member only its chunk, and a model that fits only once
  split can be loaded. ([#6232](https://github.com/tracel-ai/burn/pull/6232))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Remote execution

**Breaking:** Remote clients and servers must be upgraded together for the new protocol.
([#6002](https://github.com/tracel-ai/burn/pull/6002),
[#6010](https://github.com/tracel-ai/burn/pull/6010),
[#6144](https://github.com/tracel-ai/burn/pull/6144))

- Support quantized tensors on the remote backend and transfers larger than 1 GiB through segmented
  messages. ([#6010](https://github.com/tracel-ai/burn/pull/6010),
  [#6002](https://github.com/tracel-ai/burn/pull/6002))
- Allow reconnecting after a remote server restart and prevent transfers between independent clients
  from receiving the wrong tensor. Tensors from an ended session remain invalid.
  ([#5951](https://github.com/tracel-ai/burn/pull/5951),
  [#6145](https://github.com/tracel-ai/burn/pull/6145))
- Speed up remote tensor transfers and batched reads.
  ([#5983](https://github.com/tracel-ai/burn/pull/5983),
  [#5989](https://github.com/tracel-ai/burn/pull/5989),
  [#6017](https://github.com/tracel-ai/burn/pull/6017))
- Fix router and remote servers panicking when an integer tensor is cast to another integer
  element type.

### Training and modules

#### More flexible `#[derive(Module)]`

`#[derive(Module)]` now supports generic fields, associated types, and recursive modules. Trait
bounds are inferred from the actual field types, and generics can be shared between module fields
and skipped configuration or caches. This makes reusable module definitions easier to write.
([#6059](https://github.com/tracel-ai/burn/pull/6059))

#### Dataloader progress and training completion

**Breaking:** Progress counts samples in yielded batches, independently of iterator exhaustion.
Training no longer treats reaching the reported total as completion, supporting custom batching
strategies that buffer or drop samples and loaders with unknown output counts.

Replace `Progress::items_total` with `total: Option<usize>` (`None` means unknown), and use iterator
exhaustion instead of `Progress::is_completed()`. Custom loaders and batching strategies can
implement `expected_num_items()` to report their output sample count; `DataLoader::num_items()`
describes the input dataset size.

Split-start events now carry `Progress`. Update terminal event matches to
`EndSplit { epoch, progress }` and `EndTest { progress }`. Progress logger callbacks for starting,
updating, and ending splits now receive `&Progress`, including final snapshots for empty,
interrupted, or failed splits. ([#6184](https://github.com/tracel-ai/burn/pull/6184))

For example, update custom progress logger implementations:

```diff
- fn end_split(&mut self);
+ fn end_split(&mut self, progress: &Progress);

- fn end_test(&mut self);
+ fn end_test(&mut self, progress: &Progress);
```

- Flush partial gradient accumulation only after normal dataloader exhaustion. An interruption or
  error observed before that update discards the pending window.
  ([#6184](https://github.com/tracel-ai/burn/pull/6184))

#### Dataloader prefetch

A multi-threaded dataloader now queues at most two finished batches per worker ahead of the
consumer by default, down from a fixed 100. A batcher that builds batches on a device keeps every
queued batch resident there, which could exhaust the memory of a small GPU during training. Set the
queue with `DataLoaderBuilder::prefetch(n)` or `MultiThreadDataLoader::with_prefetch(n)`: lower it
when device memory is tight, raise it when batches are slow to build, or use `0` to hand each batch
over directly. An explicit `n` is the total queue capacity, shared by all workers; only the default
scales with their number. With four workers, the default queues eight batches, while `prefetch(2)`
queues two. Each worker also holds the batch it is waiting to send.
([#6252](https://github.com/tracel-ai/burn/pull/6252))

#### Stateful feed-forward activations

Stateful `PositionWiseFeedForward` activations such as PReLU and SwiGLU are now trained and
persisted. Older checkpoints lack these activation parameters; retain their initialized values when
loading those checkpoints by filtering to `linear_inner` and `linear_outer`.
([#5932](https://github.com/tracel-ai/burn/pull/5932))

#### Modules, losses, and metrics

- Reduce classification metric memory use by storing confusion codes as U8 on devices that support
  it, retaining the default integer dtype otherwise.
  ([#6224](https://github.com/tracel-ai/burn/pull/6224))
  [@Mikyx-1](https://github.com/Mikyx-1)
- Avoid retaining temporary QR buffers after orthogonal parameter initialization.
  ([#6172](https://github.com/tracel-ai/burn/pull/6172))
  [@EthanAlmloff](https://github.com/EthanAlmloff) [@laggui](https://github.com/laggui)
- Add `MultiHeadAttention::forward_cache_context` for faster cached inference when attention weights
  are not needed. ([#6125](https://github.com/tracel-ai/burn/pull/6125))
- Add `AvgPool3d`, `MaxPool3d`, `LpPool1d`, and `LpPool2d` pooling modules.
  ([#5801](https://github.com/tracel-ai/burn/pull/5801),
  [#5977](https://github.com/tracel-ai/burn/pull/5977))
- Add the Matthews correlation coefficient metric for binary and multiclass classification.
  ([#5818](https://github.com/tracel-ai/burn/pull/5818))
- Reject incompatible devices and dtypes in convolution, linear, normalization, and attention
  operations before backend execution, including mismatches introduced by dequantization.
  Packed quantized linear weights remain supported.
  ([#6225](https://github.com/tracel-ai/burn/pull/6225)) [@onenewcode](https://github.com/onenewcode)
- Improve half-precision CTC loss and transposed-convolution accuracy. Fix CTC execution through
  router and remote backends and reject CTC inputs exceeding the device shared-memory limit.
  ([#6103](https://github.com/tracel-ai/burn/pull/6103),
  [#5963](https://github.com/tracel-ai/burn/pull/5963),
  [#6001](https://github.com/tracel-ai/burn/pull/6001),
  [#6029](https://github.com/tracel-ai/burn/pull/6029))
- Return zero attention for fully masked rows instead of producing NaNs.
  ([#6039](https://github.com/tracel-ai/burn/pull/6039))
- Correct accelerated attention results for padding masks broadcast across query positions.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))
- Correct max-pooling SAME padding with dilation. Reject unsupported asymmetric
  deformable-convolution padding and invalid binary-cross-entropy inputs.
  ([#6023](https://github.com/tracel-ai/burn/pull/6023),
  [#6101](https://github.com/tracel-ai/burn/pull/6101),
  [#5969](https://github.com/tracel-ai/burn/pull/5969))
- Keep LBFGS parameter updates aligned when some parameters have no gradients, including when
  optimizer history exists. ([#6121](https://github.com/tracel-ai/burn/pull/6121))
- Reject incomplete LPIPS pretrained weights instead of silently using random parameters, and
  correct the AlexNet checkpoint. Use `try_init_pretrained` or `try_load_pretrained_weights` to
  handle loading failures without panicking. ([#6067](https://github.com/tracel-ai/burn/pull/6067))
- Default the F-beta metric to F1 and require a finite, positive beta. Correct CER padding handling
  and WER calculations when reference and prediction sequence lengths differ.
  ([#5965](https://github.com/tracel-ai/burn/pull/5965),
  [#5945](https://github.com/tracel-ai/burn/pull/5945),
  [#6028](https://github.com/tracel-ai/burn/pull/6028))
- Correct BLEU and ROUGE-L scores and prevent panics when prediction and reference sequence lengths
  differ. ([#6242](https://github.com/tracel-ai/burn/pull/6242))
  [@onenewcode](https://github.com/onenewcode)
- Fix `BleuScore` running and final values to use corpus-wide statistics instead of averaging batch
  scores. ([#6240](https://github.com/tracel-ai/burn/pull/6240))
  [@Arc000007](https://github.com/Arc000007)
- Fix CPU non-maximum-suppression results and restore GPU acceleration for connected components.
  ([#6019](https://github.com/tracel-ai/burn/pull/6019),
  [#6003](https://github.com/tracel-ai/burn/pull/6003))

### Tensors and backends

#### NdArray and LibTorch backend removal

**Breaking:** The deprecated NdArray and LibTorch backends have been removed. Replace `NdArray` or
`Tch` and their feature flags with a supported backend, such as `Flex` for CPU execution.
([#5999](https://github.com/tracel-ai/burn/pull/5999),
[#6009](https://github.com/tracel-ai/burn/pull/6009))

#### Tensor operations

- **Breaking:** Remove `burn_std::quantization::pack_i8s_to_u32s`. For manual word packing, pad bytes
  to groups of four and use `u32::from_le_bytes`.
  ([#6118](https://github.com/tracel-ai/burn/pull/6118)) [@onenewcode](https://github.com/onenewcode)
- **Breaking:** Reject malformed quantized `TensorData` during construction and deserialization,
  including payloads missing required padding or scales. Use `try_from_bytes` to handle errors.
  ([#5978](https://github.com/tracel-ai/burn/pull/5978),
  [#6004](https://github.com/tracel-ai/burn/pull/6004))
- Support every symmetric quantization scheme on Flex, including FP8 and FP4, while preserving the
  requested scheme when reading tensor data. E2M1 quantization maps NaN to +6, matching CUDA.
  ([#6004](https://github.com/tracel-ai/burn/pull/6004),
  [#6024](https://github.com/tracel-ai/burn/pull/6024))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)
- Add `Tensor::round_to(decimals)` for rounding to a specified number of decimal places.
  ([#5842](https://github.com/tracel-ai/burn/pull/5842))
- Add dilation to average pooling with `AvgPoolOptions::with_dilation`, matching max pooling.
  CubeCL backends use CubeK kernels for dilated 1D/2D average pooling, preserving the NHWC layout
  optimization. `count_include_pad` counts the dilated taps that land in padding. **Breaking:** Add
  `dilation: [1; N]` to `AvgPoolOptions` struct literals. **Breaking for third-party backends:**
  The `ModuleOps` average pooling methods and their backward passes take a `dilation` argument after
  `padding`, as do `module::avg_pool2d_backward` and `module::avg_pool3d_backward`.
  ([#6235](https://github.com/tracel-ai/burn/pull/6235))
  [@ashokram-si](https://github.com/ashokram-si)
- Support all indexing update operations for boolean `scatter` and `select_assign`, and add boolean
  `scatter_nd` and `gather_nd`. `Add`/`Max` use logical OR; `Mul`/`Min` use logical AND.
  **Breaking for third-party backends:** In `BoolTensorOps`, replace `bool_scatter_or` and
  `bool_select_or` with `bool_scatter` and `bool_select_assign`, respectively, accepting an
  `IndexingUpdateOp` argument and implementing each update operation. Implement the new required
  `bool_scatter_nd` and `bool_gather_nd` methods.
  ([#6035](https://github.com/tracel-ai/burn/pull/6035),
  [#6047](https://github.com/tracel-ai/burn/pull/6047))
- Add `Device::memory_capacity()` to query the runtime-reported capacity in bytes for workload
  sizing. The value is distinct from currently free memory; backends without a reported capacity
  return `None`. ([#6250](https://github.com/tracel-ai/burn/pull/6250))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)
- Add dtype-aware `TensorData::random_dtype`, `zeros_dtype`, and `ones_dtype` constructors.
  ([#6088](https://github.com/tracel-ai/burn/pull/6088))
- Add `Tolerance::get_relative()` and `get_absolute()` accessors.
  ([#6087](https://github.com/tracel-ai/burn/pull/6087)) [@laggui](https://github.com/laggui)
- Add `CaptureScope::input`, `CaptureScope::complete_with`, and `Tensor::capture_id` helpers for
  declaring graph inputs and outputs. ([#5990](https://github.com/tracel-ai/burn/pull/5990))
- Add `Graph::capture_error`, which reports why a graph replays by re-running its closure, for
  example an allocation that grew the memory pool inside the capture window.
  ([#6255](https://github.com/tracel-ai/burn/pull/6255))

#### Fusion and correctness fixes

- Fix `chunk` panicking when the requested number of chunks cannot all be filled. It now returns
  fewer chunks, as documented; for example, 6 elements in 4 chunks give 3 chunks of 2.
  ([#6263](https://github.com/tracel-ai/burn/pull/6263), [@b84330808](https://github.com/b84330808))
- Return NaN for variance reductions over empty axes instead of panicking or returning zero.
  ([issue #6236](https://github.com/tracel-ai/burn/issues/6236))
- **Breaking:** `Tensor::reshape` now treats zeros in arrays as literal zero-length dimensions,
  matching concrete `Shape` arguments. Arrays still support `-1` inference. To preserve dimension
  copying, wrap the array in `ZeroDimPlaceholders`:

  ```diff
  +use burn::tensor::ZeroDimPlaceholders;
  +
  -let y = x.reshape([0, -1]);
  +let y = x.reshape(ZeroDimPlaceholders([0, -1]));
  ```

  ONNX importers must wrap shape arrays for `allowzero=0`
  (the default) and use plain arrays for `allowzero=1`; previously generated code needs migration
  or regeneration with an updated importer. Invalid copy placeholders and ambiguous inference
  involving zero-length dimensions now produce clear reshape errors. `reshape(Shape)` and
  `unsqueeze` preserve zero-length dimensions, and mismatched target ranks are rejected.
  ([#6112](https://github.com/tracel-ai/burn/pull/6112),
  [#6257](https://github.com/tracel-ai/burn/pull/6257),
  [issue #6217](https://github.com/tracel-ai/burn/issues/6217))
  [@mattisonchao](https://github.com/mattisonchao) [@laggui](https://github.com/laggui)
- Reject `expand(Shape)` arguments whose rank differs from the output tensor rank, with a clear
  validation error. ([#6211](https://github.com/tracel-ai/burn/pull/6211))
  [@onenewcode](https://github.com/onenewcode)
- Clarify tensor type-mismatch panics with the requested type and guidance to check the rank,
  including gradient lookups.
  ([#5971](https://github.com/tracel-ai/burn/pull/5971)) [@tandede](https://github.com/tandede)
- Fix concurrent CUDA transfers between devices returning another transfer's data or deadlocking.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))
- Report lost WGPU devices instead of hanging during synchronization, and avoid Vulkan loader
  crashes during concurrent device enumeration.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))
- Fix compiler simplifications that changed NaN and signed-zero results, and invalid fused
  multiply-add generation for complex types. ([#6161](https://github.com/tracel-ai/burn/pull/6161))
- Fix half-precision `tanh` compilation with CUDA toolkits older than 12.8.
  ([#6188](https://github.com/tracel-ai/burn/pull/6188),
  [CubeCL #1856](https://github.com/tracel-ai/cubecl/pull/1856))
- Fix incorrect results and out-of-bounds accesses in fused operations with mixed vector widths,
  including quantized tensors and gather/select autotuning.
  ([#6138](https://github.com/tracel-ai/burn/pull/6138),
  [#6169](https://github.com/tracel-ai/burn/pull/6169),
  [#6091](https://github.com/tracel-ai/burn/pull/6091),
  [#6076](https://github.com/tracel-ai/burn/pull/6076))
- Fix fusion failures when tensors are dropped on another thread, and report failed transaction
  reads as errors instead of panicking. ([#6086](https://github.com/tracel-ai/burn/pull/6086),
  [#5970](https://github.com/tracel-ai/burn/pull/5970),
  [#5986](https://github.com/tracel-ai/burn/pull/5986))
- Correct GELU gradients, scalar powers with integer exponents, and floating-point remainder results
  for special values. ([#6114](https://github.com/tracel-ai/burn/pull/6114),
  [#6058](https://github.com/tracel-ai/burn/pull/6058),
  [#6045](https://github.com/tracel-ai/burn/pull/6045))
- Preserve the Flex32 dtype during tensor uploads, `TensorData` conversions, and filled-data
  construction. ([#6056](https://github.com/tracel-ai/burn/pull/6056),
  [#6062](https://github.com/tracel-ai/burn/pull/6062),
  [#6069](https://github.com/tracel-ai/burn/pull/6069))
- Return off-value rows from `one_hot_fill` for out-of-range indices, including NaN and infinity.
  ([#6065](https://github.com/tracel-ai/burn/pull/6065))
- Fix boolean `scatter_nd` reductions losing updates on CubeCL backends when indices repeat a
  target; the updates are now applied with atomic stores.
  ([#6215](https://github.com/tracel-ai/burn/pull/6215))

### Checkpoints and interoperability

#### Checkpoint adapters use `StoreTensor`

**Breaking:** Module adapters now use `StoreTensor` to preserve source tensor layouts during
checkpoint loading. Use `StoreTensor` instead of `burn_pack::Tensor` in module adapters and
`ModuleSnapshot::apply`. Convert pack tensors with `.into()`, use metadata accessors instead of
public fields, and call `into_pack()` when a pack tensor is needed.
([#6113](https://github.com/tracel-ai/burn/pull/6113))

```rust,ignore
let tensor: burn_store::StoreTensor = pack_tensor.into();
let shape = tensor.shape();
let pack_tensor = tensor.into_pack();
```

- Add `PytorchReader::from_bytes` for loading checkpoints from memory and configurable
  `burn_pack::ReaderLimits` through `Reader::with_limits`.
  ([#5755](https://github.com/tracel-ai/burn/pull/5755),
  [#6007](https://github.com/tracel-ai/burn/pull/6007))
- Speed up loading strided and transposed PyTorch checkpoint weights.
  ([#6041](https://github.com/tracel-ai/burn/pull/6041),
  [#6113](https://github.com/tracel-ai/burn/pull/6113))
- Avoid repeated SafeTensors header parsing when loading models with many tensors.
  ([#6269](https://github.com/tracel-ai/burn/pull/6269))
  [@Arc000007](https://github.com/Arc000007)

### Extensions

#### Vision

- Speed up Flex CPU median blur for F16, BF16, F32, F64, and integer tensors by selecting
  neighborhood medians without stacking the images. U8, I8, and exact normalized RGB8 inputs
  use sliding histograms for larger filters. Floating-point autodiff uses the selected source
  pixels for backward. `median` and `MedianBlur` now also accept `Tensor<4, Int>` images;
  specify `Tensor::<4>` on float constructors if their kind was previously inferred from these calls.
  ([#6219](https://github.com/tracel-ai/burn/pull/6219))
  [@Mikyx-1](https://github.com/Mikyx-1)

#### Batched linear solves

`burn::linalg::solve` solves square, nonsingular systems `A @ X = B` with vector or matrix
right-hand sides and broadcast batches. Flex and CubeCL provide optimized factorization and
substitution paths with autodiff support. F16 and BF16 inputs compute in F32 and cast back;
nonsquare or singular systems are rejected.
([#5856](https://github.com/tracel-ai/burn/pull/5856))
[@Mikyx-1](https://github.com/Mikyx-1)

#### Cholesky decomposition

`burn::linalg::cholesky(a, upper)` factors real symmetric positive-definite matrices, including
batches, with optimized Flex CPU and supported CubeCL GPU paths. Autodiff uses the portable tensor
implementation. Pass `false` for a lower triangular factor or `true` for an upper triangular factor;
only the selected input triangle is read. Non-positive or non-finite pivots are rejected.
([#5858](https://github.com/tracel-ai/burn/pull/5858))
[@Mikyx-1](https://github.com/Mikyx-1) [@laggui](https://github.com/laggui)

#### FFTs of arbitrary sizes

Real FFTs and their inverses now accept arbitrary sizes, removing the power-of-two restriction.
Complex inverse FFT (`ifft`) is also available.
([#5840](https://github.com/tracel-ai/burn/pull/5840),
[#5757](https://github.com/tracel-ai/burn/pull/5757))
