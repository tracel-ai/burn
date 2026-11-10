# Changelog

Changes for the next release. For published releases, see
[GitHub Releases](https://github.com/tracel-ai/burn/releases).

## Unreleased

<!-- Keep subject groups stable and feature subsections alphabetical within each group.
See CONTRIBUTING.md#changelog for entry guidance. -->

### Runtime and backends

#### Backend extensions

Backend extensions run on the remote backend without handwritten remote code: listing `Remote`
beside `Fusion` in `#[backend_extension]` generates the client, which sends each call as a custom
operation's IR, and `<Trait>Handlers::register` for the server's `CustomOpRegistry`. Ordinary
arguments that are not primitives take `#[fusion(scalar = .., decode = ..)]`, and extension values
opt in with `#[extension_type(fusion, remote)]`. `CustomOpRegistry::try_register` refuses a
handler from another `HandlerSource` under an id already taken.
([#6308](https://github.com/tracel-ai/burn/pull/6308))

**API migration:** a `#[backend_extension]` Fusion operation's default id is now
`"Trait::method"` instead of the method name, so two traits' same-named methods no longer share
an id. A custom optimizer matching a default id must match the new one, or the method can keep its
old id with `#[fusion(id = "method")]`. An extension listing both `Fusion` and `Remote` now
generates its `BackendRouter<_>` implementation, so a handwritten `impl Trait for BackendRouter<_>`
conflicts with it (E0119); delete the handwritten one.
([#6308](https://github.com/tracel-ai/burn/pull/6308))

- Backend extensions run on a sharded compute device: list `ShardedCompute` beside `Fusion` in
  `#[backend_extension]` and pass the handlers with `ShardOptions::with_custom_ops`. A custom op
  gathers its inputs, unless a `CustomOpRule` splits it like a built-in op of a given kind or runs it
  once over every member through a `CollectiveOpRegistry` handler. `ShardedComputeDevice::members`
  lists the members' devices, for an extension that checks what their hardware supports.
  ([#6395](https://github.com/tracel-ai/burn/pull/6395))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- **Breaking (experimental):** `ShardedComputeDevice::with_member_settings` and
  `DispatchDevice::shard` take the device's custom ops: pass `ShardCustomOps::default()` for none.
  `ShardError` has a new `CustomOpsBackend` variant.
  ([#6395](https://github.com/tracel-ai/burn/pull/6395))

#### Compilation and autotuning

- Include the preceding `CompileOnly` pass's statistics in the following
  `CompileAndAutotune` pass on CubeCL backends, so deferred completions appear in its snapshot.

- Compile autotuning candidates in parallel, including CUDA binary assembly, to reduce startup time.
  ([#5996](https://github.com/tracel-ai/burn/pull/5996),
  [#6074](https://github.com/tracel-ai/burn/pull/6074))

- Reuse compiled CPU kernels across process runs when persistent compilation caching is enabled.
  ([#6099](https://github.com/tracel-ai/burn/pull/6099))

- Share autotuning results across devices with the same hardware. Previously saved per-device
  results are tuned again once. ([#6150](https://github.com/tracel-ai/burn/pull/6150))

- Stop dumping LLVM compiler IR into normal experiment logs, reducing compilation overhead and log
  size. ([#6188](https://github.com/tracel-ai/burn/pull/6188),
  [CubeCL #1895](https://github.com/tracel-ai/cubecl/pull/1895))

- Fix compiler simplifications that changed NaN and signed-zero results, and invalid fused
  multiply-add generation for complex types. ([#6161](https://github.com/tracel-ai/burn/pull/6161))

- Fix half-precision `tanh` compilation with CUDA toolkits older than 12.8.
  ([#6188](https://github.com/tracel-ai/burn/pull/6188),
  [CubeCL #1856](https://github.com/tracel-ai/cubecl/pull/1856))

#### CubeK tile kernels

- Run matmul, fused matmul and attention on CubeCL backends through CubeK's tile routines, chosen
  by CubeK's autotune or, without the `autotune` feature, by its rules. Quantized matmul inputs are
  read packed where the routines can, and dequantized first elsewhere. Attention takes a bool mask,
  a custom scale and both causal alignments; an additive bias or a softcap runs the decomposed
  attention. Without `autotune`, a fused matmul runs its matmul and element-wise block as separate
  kernels. The multi-level matmul and flash attention paths are removed with the experimental
  `tile-kernels-only` feature. Migration: `MatmulStrategy::Cube` now names CubeK's rule-based
  choice; `AttentionStrategy` keeps only `Autotune` and `Fallback`, and `attention` returns the
  tensor rather than a `Result`; `matmul_autotune`, `flash_attention` and
  `flash_unsupported_options` are gone (use `matmul` and `attention`).
  ([#6193](https://github.com/tracel-ai/burn/pull/6193),
  [#6199](https://github.com/tracel-ai/burn/pull/6199),
  [#6246](https://github.com/tracel-ai/burn/pull/6246),
  [#6273](https://github.com/tracel-ai/burn/pull/6273)) [@louisfd](https://github.com/louisfd)

#### Device descriptions and measurements

**Breaking:** What a device is and what it measures are `Backend` methods in Burn's own types, so a
remote device answers them from its server. `Device::description()` states what the driver says
without opening the device; `measure_telemetry()`, `measure_memory()` and `measure_peaks(&keys)`
run something — read the card's counters, look through the allocator, launch probes. A cubecl
device's features, limits and compile target are `Device::capabilities()`, answered by the server
for a remote device.

```diff
-device.memory_pool_usage()  /  device.memory_pool_report()
+device.measure_memory()       // .usage, .pools
-device.memory_capacity()
+device.description()          // .max_memory
-device.performance_stats(&[ThroughputKey { mode: ThroughputMode::memory(MemoryAccess::Copy) }])
+device.measure_peaks(&[PeakKey::Memory(MemoryAccess::Copy)])   // Peak { per_second }
-device.identity()
+device.capabilities()         // .properties.identity
```

Third-party backends replace `memory_pool_report`, `memory_pool_usage` and `memory_capacity`
with `measure_memory`, and may implement `device_description`, `measure_telemetry` and
`measure_peaks`.

- Report the runtime-stated capacity in bytes for workload sizing, now
  `Device::description()`'s `max_memory`. The value is distinct from currently free memory;
  backends without a reported capacity return `None`. ([#6250](https://github.com/tracel-ai/burn/pull/6250))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Device transfers

- Fix concurrent CUDA transfers between devices returning another transfer's data or deadlocking.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))

- Fix CPU and WGPU reads on one stream returning a tensor's old values while another stream's
  in-place write to it was still queued. ([#6332](https://github.com/tracel-ai/burn/pull/6332),
  [CubeCL #2002](https://github.com/tracel-ai/cubecl/pull/2002))

- Fix bool `to_device` between backends panicking when the target uses a different bool storage,
  such as Flex to CUDA or Vulkan. Transfers preserve source bool storage when supported by the
  destination and otherwise use its configured bool storage. CubeCL converts on a device that
  supports both representations and normalizes every nonzero true value to 1. Fusion and remote
  transfers report the resulting storage dtype.
  **Breaking for third-party backends:** `BoolTensorOps::bool_to_device` and
  `MultiBackendBridge::change_backend_bool` now take an `out_dtype: BoolDType` argument.
  `RouterChannel::change_client_backend` takes a destination `DType`. Resolve the destination
  storage at the caller and forward it through wrappers to avoid intermediate conversions.
  ([issue #6389](https://github.com/tracel-ai/burn/issues/6389),
  [issue #6391](https://github.com/tracel-ai/burn/issues/6391))

#### Device-time instrumentation

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

- A sharded compute device whose members fuse is instrumented on every member: each kernel is
  recorded under its component and the run reads fused. `DispatchDevice::fusion_worker` is replaced
  by `fusion_workers`, one id per member.
  ([#6430](https://github.com/tracel-ai/burn/pull/6430))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- `Breakdown::per_member` holds each member's device and idle time on a sharded compute device
  whose members fuse, and the time of kernels launched off the members' workers. Its report prints
  them, and the clean run beyond the slowest member's kernels, in place of a host split.
  Measurement files are version 5. ([#6430](https://github.com/tracel-ai/burn/pull/6430))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Flex SIMD operations

- Speed up Flex CPU F32 `exp`, `sigmoid`, and `softmax` with a SIMD `exp` within 2 ulp of libm, and
  run F32/F64 `round`, `floor`, `ceil`, and `trunc` with SIMD rounding. Flex `round` now returns
  `-0.0` for `-0.5` on every layout, as IEEE ties-to-even does. F32 `log`, `tanh`, `erf`, `gelu`,
  and the GELU backward pass also run with SIMD; `log` matches the `libm` crate bit for bit, `tanh`
  is within 2 ulp of libm, and `erf` within 1 ulp. F32 `log1p`, `atanh`, `log_sigmoid`, and the
  log-sigmoid backward pass also run with SIMD; `log1p` and `atanh` match the `libm` crate bit for
  bit. F32 `sin` and `cos` run with SIMD within 2 ulp of libm, and `tan` within 4; `atan`, `asin`,
  and `acos` within 3, 2, and 1 ulp. F32 `atan2` (within 3 ulp) and `hypot` (within 1 ulp) run with
  SIMD on same-shape contiguous operands, and the GELU and log-sigmoid backward passes run in one
  pass. F32 `sinh` and `cosh` run with SIMD within 2 ulp of libm.
  ([#6313](https://github.com/tracel-ai/burn/pull/6313),
  [#6341](https://github.com/tracel-ai/burn/pull/6341),
  [#6345](https://github.com/tracel-ai/burn/pull/6345),
  [#6347](https://github.com/tracel-ai/burn/pull/6347),
  [#6352](https://github.com/tracel-ai/burn/pull/6352),
  [#6353](https://github.com/tracel-ai/burn/pull/6353),
  [#6354](https://github.com/tracel-ai/burn/pull/6354)) [@antimora](https://github.com/antimora)

- Speed up Flex CPU `clamp` and `sign` by letting their loops vectorize, and run F32/F64 `powi`
  with integer exponents other than 0, ±1 and ±2 with SIMD, matching the scalar results bit for
  bit. ([#6351](https://github.com/tracel-ai/burn/pull/6351)) [@antimora](https://github.com/antimora)

#### Flex tensor allocation

- Halve peak memory and speed up Flex CPU `zeros` and `repeat_dim` by allocating the output once.
  ([#6384](https://github.com/tracel-ai/burn/pull/6384))

#### Fusion

- Reduce fusion's CPU overhead by skipping unnecessary reduction-fuser resets.
  ([#6036](https://github.com/tracel-ai/burn/pull/6036)) [@louisfd](https://github.com/louisfd)

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

#### Graph capture

- Add `CaptureScope::input`, `CaptureScope::complete_with`, and `Tensor::capture_id` helpers for
  declaring graph inputs and outputs. ([#5990](https://github.com/tracel-ai/burn/pull/5990))

- Add `Graph::capture_error`, which reports why a graph replays by re-running its closure, for
  example an allocation that grew the memory pool inside the capture window.
  ([#6255](https://github.com/tracel-ai/burn/pull/6255))

- `capture` on a remote device records and replays the graph on the server's device. One session
  captures per server device at a time. A replay that fails is the error of the next read or sync
  on its stream. ([#6308](https://github.com/tracel-ai/burn/pull/6308))

- `capture` works on a sharded compute device: each member records its own graph, collectives
  included, and a replay runs every member's. ([#6429](https://github.com/tracel-ai/burn/pull/6429))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Matrix multiplication

- Enable CUDA tensor-core acceleration on Blackwell and Blackwell Ultra GPUs, including B300.
  ([#6161](https://github.com/tracel-ai/burn/pull/6161))

- Improve CubeCL CPU matrix multiplication by selecting tile sizes that fit the vector width
  and using GEMM settings tailored to CPU hardware.
  ([#6177](https://github.com/tracel-ai/burn/pull/6177),
  [#6245](https://github.com/tracel-ai/burn/pull/6245))
  [@Charles23R](https://github.com/Charles23R)

#### Matrix multiplication precision

Matrix multiplication precision is now configurable through `[cubek.matmul].allow_tf32` in
`burn.toml`. Set it to `false` to disable automatic F32-to-TF32 conversion; the default is `true`.
([#6170](https://github.com/tracel-ai/burn/pull/6170))

```toml
# burn.toml
[cubek.matmul]
allow_tf32 = false
```

See [runtime configuration](./advanced/runtime-configuration.md#cubek-matmul-precision) for details.

#### NdArray and LibTorch backend removal

**Breaking:** The deprecated NdArray and LibTorch backends have been removed. Replace `NdArray` or
`Tch` and their feature flags with a supported backend, such as `Flex` for CPU execution.
([#5999](https://github.com/tracel-ai/burn/pull/5999),
[#6009](https://github.com/tracel-ai/burn/pull/6009))

#### WGPU device handling

- Fix incorrect Vulkan results on NVIDIA GPUs caused by shared-memory atomic operations returning
  zero. ([CubeCL #2059](https://github.com/tracel-ai/cubecl/pull/2059)) [@laggui](https://github.com/laggui)

- Report lost WGPU devices instead of hanging during synchronization, and avoid Vulkan loader
  crashes during concurrent device enumeration.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))

### Distributed execution

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

- Float collectives under autodiff run on the backend's native collective, NCCL on CUDA, with a
  backward of their own, instead of being emulated with tracked transfers between devices. A
  collective's outputs share that backward, so their losses must go through a single `backward`.
  ([#6274](https://github.com/tracel-ai/burn/pull/6274))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- Drain every device's pending fusion work at once before a collective, speeding up
  tensor-parallel decode. ([#6293](https://github.com/tracel-ai/burn/pull/6293))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- Under fusion, a collective the backend runs natively (NCCL on CUDA) is queued as one part on
  each member's own stream instead of waiting for every member first, so the caller keeps queueing
  work while the devices run it. Tensor-parallel decode over four L4 GPUs runs about 1.8x faster.
  A member whose input carries a failure still joins the others, which receive NaN (zeros for
  integers and booleans) in its place. Backend implementors provide
  `FusionBackend::collective_parts`, returning `None` to keep a collective on the caller.
  ([#6421](https://github.com/tracel-ai/burn/pull/6421))

#### Distributed data parallel training

- Fix DDP gradient-synchronization hangs across local and remote devices.
  ([#6144](https://github.com/tracel-ai/burn/pull/6144))

- Coordinate DDP batch availability before backward passes and synchronize epoch transitions.
  Workers must yield the same number of training batches; uneven exhaustion now reports an error.
  ([#6184](https://github.com/tracel-ai/burn/pull/6184))

#### Layer parallelism

Models can now be split across devices by layer, enabling training and inference with models that
exceed a single device's memory. The new `LayerParallelism` trait and `DistributedLayeredModel`
wrapper support sequential execution, transferring activations between devices.
`InferencePipeline` streams several batches through such a model with a thread per stage, so the
stages overlap, and `TrainingPipeline` trains it with the GPipe or 1F1B schedule. A signal
passed between layers derives `AutodiffSignal` to train through. `Learner` integration is not
included.
**Breaking:** `LayerParallelism` requires `from_layers`, which rebuilds the model from the
`LayerSequence` that `into_layers` returns.
([#5702](https://github.com/tracel-ai/burn/pull/5702),
[#6277](https://github.com/tracel-ai/burn/pull/6277))
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
  element type. ([#6308](https://github.com/tracel-ai/burn/pull/6308))

- Fix a HIP worker crashing when a session ends as the process exits.
  ([#6308](https://github.com/tracel-ai/burn/pull/6308))

- A read on a router or remote server of a tensor whose producing work failed reports that failure
  instead of ending the session. ([#6308](https://github.com/tracel-ai/burn/pull/6308))

- Remote and router errors follow the CubeCL and fusion model. On the server, an operation reading
  a failed tensor is skipped and a failed transfer or unknown graph claims
  the tensors it was going to write; a read of one reports the root failure while the session
  keeps running. On the client, a lost connection ends the session instead of panicking inside an
  operation: every later `flush`, `sync` and read reports why as a poisoned device
  (`ExecutionError::is_device_poisoned`), and a `to_device` across it yields a tensor whose read
  reports the same. A sharded compute device's `flush` reports a member's error instead of holding it for `sync`.
  **Breaking:** `RouterClient::flush` returns `Result<(), ExecutionError>`.
  ([#6329](https://github.com/tracel-ai/burn/pull/6329))

- Servers answer device queries registered with `with_device_query`, asked by
  `RemoteDevice::query`. **Breaking:** `ServeError::CustomOpBackend` is renamed `HandlerBackend`,
  as it also covers device queries. ([#6308](https://github.com/tracel-ai/burn/pull/6308))

- Fix a router tensor read that could find its tensor already dropped when another thread gave up
  the tensor's last other handle at the same moment, failing a sharded compute device's read with
  `no entry found for key`. ([#6405](https://github.com/tracel-ai/burn/pull/6405))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

#### Sharded compute devices

Experimental: with the `shard` feature, `Device::shard_options(&devices).init()` makes one device
whose tensors are split over `devices` (crate `burn-shard`).

- Each member's shard is made by the caller, where it lives: `Tensor::from_shards(Shards { shards,
  sharding }, &device)` joins one tensor per member, each already on its member's device
  (`Device::shard_members`), and refuses shards that do not make up one tensor with a
  `ShardsError`. `into_shards` gives the `Shards` back. A split's chunks are balanced in member
  order, and `shard::DimSplit` gives each member's range and cuts a `TensorData` to it, so a
  member reading a checkpoint reads only its chunk. The tensor's sharding is read with
  `Tensor::sharding`, and ops move shards between members themselves where their inputs need it. A
  random tensor on a sharded compute device is drawn once and replicated.
  ([#6232](https://github.com/tracel-ai/burn/pull/6232),
  [#6272](https://github.com/tracel-ai/burn/pull/6272),
  [#6397](https://github.com/tracel-ai/burn/pull/6397))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- Attention split on its batch or heads, and `swap_dims`, `permute`, `expand`, `slice` and
  `slice_assign` on a sharded compute device keep their input's split instead of gathering it, so a
  tensor-parallel transformer layer communicates only in its all-reduces.
  ([#6372](https://github.com/tracel-ai/burn/pull/6372))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- Elementwise ops, reductions, and softmax, cumsum, sort and topk on a sharded compute device keep
  their input's split or partial sum where the math allows, instead of gathering it.
  ([#6387](https://github.com/tracel-ai/burn/pull/6387))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

- A quantized tensor splits over a sharded compute device, its codes and scales cut as they are,
  so a quantized model's weights split for tensor parallelism. A split must give every member whole
  quantization blocks, and whole packed words along the packed axis, which `from_shards` checks. **Breaking:**
  `ShardingError::Quantized` is replaced by `ShardingError::QuantizedSplit`, returned for a split
  that would cut either. Split shards quantized each on their own must share a per-tensor or global
  scale: a read or an op that gathers them otherwise fails with `ShardingError::QuantizedScales`,
  and `DimSplit::whole_data` returns a `Result`. Reading a failed sharded operation returns its
  recorded execution error instead of panicking.
  ([#6396](https://github.com/tracel-ai/burn/pull/6396),
  [#6408](https://github.com/tracel-ai/burn/pull/6408))
  [@ThierryCantin-Demers](https://github.com/ThierryCantin-Demers)

### Tensors

#### Boolean indexing

- Support all indexing update operations for boolean `scatter` and `select_assign`, and add boolean
  `scatter_nd` and `gather_nd`. `Add`/`Max` use logical OR; `Mul`/`Min` use logical AND.
  **Breaking for third-party backends:** In `BoolTensorOps`, replace `bool_scatter_or` and
  `bool_select_or` with `bool_scatter` and `bool_select_assign`, respectively, accepting an
  `IndexingUpdateOp` argument and implementing each update operation. Implement the new required
  `bool_scatter_nd` and `bool_gather_nd` methods.
  ([#6035](https://github.com/tracel-ai/burn/pull/6035),
  [#6047](https://github.com/tracel-ai/burn/pull/6047))

- Fix boolean `scatter_nd` reductions losing updates on CubeCL backends when indices repeat a
  target; the updates are now applied with atomic stores.
  ([#6215](https://github.com/tracel-ai/burn/pull/6215))

#### Flex32 tensor data

- Preserve the Flex32 dtype during tensor uploads, `TensorData` conversions, and filled-data
  construction. ([#6056](https://github.com/tracel-ai/burn/pull/6056),
  [#6062](https://github.com/tracel-ai/burn/pull/6062),
  [#6069](https://github.com/tracel-ai/burn/pull/6069))

#### Quantization

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

- Run quantized `argmax`, `argmin`, `gather` and `quantize_dynamic` on the backend's own
  implementations through device dispatch instead of dequantizing first.
  ([#6383](https://github.com/tracel-ai/burn/pull/6383))

#### Random operation keys

Random operations now reserve a `RandomKey` when requested, before shard, fusion, or remote queues
execute them. Even a draw dropped before execution consumes one key. Reseeding leaves pending
draws unchanged, and sharded tensors no longer depend on fusion worker scheduling. Seeded sequences change from previous versions; equality across devices
or different shardings is not guaranteed.

Backend implementors must provide `Backend::next_random_key` and accept a `RandomKey` in
`float_random` and `int_random`. Sampling must use only the supplied key, without advancing the
implicit generator. `Tensor::random` and `random_like` reserve keys at the tensor bridge;
direct backend callers must supply a key. Use `RandomKey::split` to derive child keys.
Router clients must also implement `next_random_key`; random-operation IR now carries a key
scalar, so remote clients and servers must be upgraded together (protocol version 12).
Remote clients reserve keys locally without waiting for queued service work.

Capture preserves explicit backend keys. Implicit random operations (`Tensor::random` and
`random_like`) during capture are unsupported and panic; capture does not allocate seeds or
own a random generator.

#### Tensor chunking

- Fix `chunk` panicking when the requested number of chunks cannot all be filled. It now returns
  fewer chunks, as documented; for example, 6 elements in 4 chunks give 3 chunks of 2.
  ([#6263](https://github.com/tracel-ai/burn/pull/6263), [@b84330808](https://github.com/b84330808))

#### Tensor data constructors

- Add dtype-aware `TensorData::random_dtype`, `zeros_dtype`, and `ones_dtype` constructors.
  ([#6088](https://github.com/tracel-ai/burn/pull/6088))

#### Tensor expansion

- Reject `expand(Shape)` arguments whose rank differs from the output tensor rank, with a clear
  validation error. ([#6211](https://github.com/tracel-ai/burn/pull/6211))
  [@onenewcode](https://github.com/onenewcode)

#### Tensor gathering and scattering

- Allow `gather` and `scatter` indices smaller than the tensor on dimensions other than `dim`
  (`indices.shape[d] <= tensor.shape[d]`), like PyTorch; `scatter` updates only the covered
  positions. CubeCL also fuses these gathers with compatible element-wise operations.
  Larger indices are still rejected. Third-party backends must accept these shapes in
  `*_gather` and `*_scatter`. ([#6381](https://github.com/tracel-ai/burn/pull/6381))

#### Tensor one-hot encoding

- Return off-value rows from `one_hot_fill` for out-of-range indices, including NaN and infinity.
  ([#6065](https://github.com/tracel-ai/burn/pull/6065))

#### Tensor powers

- Fix integer `powi` and `powi_scalar` returning wrapped values for negative exponents on Flex and
  CubeCL backends. They now give `1 / base^|exponent|` truncated toward zero: 0 unless the base is
  1 or -1. ([#6355](https://github.com/tracel-ai/burn/pull/6355),
  [@b84330808](https://github.com/b84330808))

- Correct scalar powers with integer exponents.
  ([#6058](https://github.com/tracel-ai/burn/pull/6058))

#### Tensor reductions

- Autotune CubeCL `sum_dims` between folded reductions and a copy followed by one reduction,
  including the copy cost. Tuning anchors shapes and strides to reuse plan choices across nearby
  workloads; Full autotuning keeps exact shapes and strides.
  ([#6376](https://github.com/tracel-ai/burn/pull/6376))
  [@EthanAlmloff](https://github.com/EthanAlmloff)

- Return NaN for variance reductions over empty axes instead of panicking or returning zero.
  ([issue #6236](https://github.com/tracel-ai/burn/issues/6236))

- Fix f16 min/max and top-k operations with indices failing during autotuning on CubeCL backends.
  ([#6428](https://github.com/tracel-ai/burn/pull/6428)) [@laggui](https://github.com/laggui)

#### Tensor remainder

- Correct floating-point remainder results for special values.
  ([#6045](https://github.com/tracel-ai/burn/pull/6045))

#### Tensor reshaping

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

#### Tensor rounding

- Add `Tensor::round_to(decimals)` for rounding to a specified number of decimal places.
  ([#5842](https://github.com/tracel-ai/burn/pull/5842))

#### Tensor slicing

- Allow more CubeCL slices to share their source buffer, avoiding copies of KV-cache prefixes.
  ([#6136](https://github.com/tracel-ai/burn/pull/6136))
  [@ssoj13](https://github.com/ssoj13) [@laggui](https://github.com/laggui)

#### Tensor type diagnostics

- Clarify tensor type-mismatch panics with the requested type and guidance to check the rank,
  including gradient lookups.
  ([#5971](https://github.com/tracel-ai/burn/pull/5971)) [@tandede](https://github.com/tandede)

#### Tolerance accessors

- Add `Tolerance::get_relative()` and `get_absolute()` accessors.
  ([#6087](https://github.com/tracel-ai/burn/pull/6087)) [@laggui](https://github.com/laggui)

#### Zero-sized tensor operations

- Fix in-place CubeCL operations panicking in debug builds on tensors with a zero-sized dimension.
  ([#6348](https://github.com/tracel-ai/burn/pull/6348))

### Modules

#### Attention

Grouped-query and sliding-window attention now work across backends and autodiff, with native
CubeCL acceleration. Query heads can share fewer key/value heads; invalid head counts and a
zero-length sliding window are rejected.

**Breaking:** `AttentionModuleOptions` gains `sliding_window: Option<u32>`. Add
`sliding_window: None` to struct literals to preserve existing behavior, or use
`..Default::default()`. The window is always bottom-right aligned; `Some(0)` is rejected, use
`None` to disable it.

**Breaking:** `AttentionModuleOptions` gains `causal_alignment`. Add
`causal_alignment: CausalAlignment::BottomRight` to struct literals to preserve existing behavior,
or use `..Default::default()`. Select `TopLeft` for causal masks aligned to the first query/key
positions when sequence lengths differ.
([#6127](https://github.com/tracel-ai/burn/pull/6127),
[#5953](https://github.com/tracel-ai/burn/pull/5953))
[@gzsombor](https://github.com/gzsombor) [@Andy2887](https://github.com/Andy2887)
[@louisfd](https://github.com/louisfd)

- Run attention's backward pass on CubeCL backends as CubeK kernels from the forward's saved
  logsumexp, rather than autodiff over the decomposed attention. The kernels take f32, f16 and
  bf16 attention without a mask, bias or softcap, causal or not; a softcap, a top-left causal
  mask over sequences of different lengths or an element the kernels do not take is composed from
  tensor operations, and a mask or an additive bias keeps the decomposed gradients. Autotune
  may pick a routine that sums the query gradient with atomic adds, so its last bits can differ
  from run to run. Backends can provide their own through `ModuleOps::attention_with_logsumexp`
  and `ModuleOps::attention_backward`. [@louisfd](https://github.com/louisfd)

- Add `MultiHeadAttention::forward_cache_context` for faster cached inference when attention weights
  are not needed. ([#6125](https://github.com/tracel-ai/burn/pull/6125))

- Return zero attention for fully masked rows instead of producing NaNs.
  ([#6039](https://github.com/tracel-ai/burn/pull/6039))

- Correct accelerated attention results for padding masks broadcast across query positions.
  ([#6008](https://github.com/tracel-ai/burn/pull/6008))

#### Attention over a key/value cache

`burn::tensor::module::attention` takes an optional `AttentionCache`: the key and value are then a
cache held at a fixed capacity, each sequence attending its first `lengths[b]` positions, with the
queries as the last attended ones. The cache is contiguous, `[batch, kv_heads, capacity, dim]`
(`AttentionCache::new(lengths)`), or paged: pools of pages and a `[batch, pages_per_sequence]`
table naming each sequence's pages (`AttentionCache::paged(lengths, pages)`). Since the lengths are
read on the device, a decode keeps the same shapes at every step. Causal masking and the sliding
window apply at each query's own position, and a mask or an additive bias spans the capacity.

On CubeCL backends it runs cubek's tile routines, the same ones as an attention without a cache
(with a softcap, a bias, or a mask over a cache, it composes from tensor operations, as other
backends do). Autodiff composes it from tensor operations.

**Breaking:** `attention` and `ModuleOps::attention` take the cache as a new argument before
`options`; pass `None` for an attention over the key and value as given. `AttentionOpIr` gains
`lengths` and `pages`, so remote clients and servers must be upgraded together (protocol
version 12).

#### Convolution

- Avoid a full-sized gradient copy in CubeCL FP32 `conv2d` bias backward, while retaining
  single-pass accumulation for FP16/BF16 to avoid intermediate overflow.
  ([#6173](https://github.com/tracel-ai/burn/pull/6173))
  [@EthanAlmloff](https://github.com/EthanAlmloff)

- Speed up CubeCL `conv_transpose3d` and reduce Flex `conv2d` memory use and layout conversions.
  ([#5793](https://github.com/tracel-ai/burn/pull/5793),
  [#6075](https://github.com/tracel-ai/burn/pull/6075))

- Add accelerated split convolution weight-gradient autotuning candidates on CubeCL backends,
  retaining F32 partial sums for F16 and BF16 inputs. Cached candidates fall back when a shape
  exceeds the split workspace limit.

- Add F32 unit-convolution and im2col autotuning candidates on CubeCL backends, including
  split depthwise weight gradients. Forward im2col limits its column workspace to 128 MiB.
  ([#6048](https://github.com/tracel-ai/burn/pull/6048)) [@znicelya](https://github.com/znicelya)

- Avoid excessive kernel compilation for large depthwise-convolution filters, regardless of
  channel count.
  ([#6072](https://github.com/tracel-ai/burn/pull/6072),
  [#6048](https://github.com/tracel-ai/burn/pull/6048))

- Correct `Same` padding for dilated 1D, 2D, and 3D convolutions, including Conv3d configurations
  whose effective kernel permits symmetric padding.
  ([#6286](https://github.com/tracel-ai/burn/pull/6286)) [@onenewcode](https://github.com/onenewcode)

- Accept asymmetric padding in `conv3d`, and stop `Conv3d` with `PaddingConfig3d::Same` from
  panicking when the padding is asymmetric, for example with a stride above 1.
  ([#4837](https://github.com/tracel-ai/burn/issues/4837), [#6289](https://github.com/tracel-ai/burn/pull/6289))

- Improve half-precision transposed-convolution accuracy.
  ([#5963](https://github.com/tracel-ai/burn/pull/5963))

- Reject unsupported asymmetric deformable-convolution padding.
  ([#6101](https://github.com/tracel-ai/burn/pull/6101))

#### GELU and sigmoid

- Run GELU and sigmoid forward and backward operations in single CubeCL kernels.
  ([#6114](https://github.com/tracel-ai/burn/pull/6114)) [@ssoj13](https://github.com/ssoj13)

- Correct GELU gradients.
  ([#6114](https://github.com/tracel-ai/burn/pull/6114))

#### Interpolation options

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

#### Linear-algebra layers behind `linalg`

**Breaking:** `burn` and `burn-nn` no longer build `burn-linalg` unless the `linalg` feature is
enabled. `CosineSimilarity`, `PairwiseDistance`, `CosineEmbeddingLoss`, `TripletMarginLoss`, and
`Initializer::Orthogonal` are now behind it. Add `linalg` to your `burn` features (or `burn-nn`
features) to keep using them. ([#6327](https://github.com/tracel-ai/burn/pull/6327))

#### Module derivation

`#[derive(Module)]` now supports generic fields, associated types, and recursive modules. Trait
bounds are inferred from the actual field types, and generics can be shared between module fields
and skipped configuration or caches. This makes reusable module definitions easier to write.
([#6059](https://github.com/tracel-ai/burn/pull/6059))

Enum variants can also hold named fields, such as `Wide { first: Linear, second: Linear }`. Their
tensors are saved under `Wide.first` and `Wide.second`, the same paths as a tuple variant wrapping
a struct with those fields. ([#6388](https://github.com/tracel-ai/burn/pull/6388))

#### Module input validation

- Reject incompatible devices and dtypes in convolution, linear, normalization, and attention
  operations before backend execution, including mismatches introduced by dequantization.
  Packed quantized linear weights remain supported.
  ([#6225](https://github.com/tracel-ai/burn/pull/6225)) [@onenewcode](https://github.com/onenewcode)

#### Orthogonal initialization

- Avoid retaining temporary QR buffers after orthogonal parameter initialization.
  ([#6172](https://github.com/tracel-ai/burn/pull/6172))
  [@EthanAlmloff](https://github.com/EthanAlmloff) [@laggui](https://github.com/laggui)

#### Pooling

- Add `AvgPool3d`, `MaxPool3d`, `LpPool1d`, and `LpPool2d` pooling modules.
  ([#5801](https://github.com/tracel-ai/burn/pull/5801),
  [#5977](https://github.com/tracel-ai/burn/pull/5977))

- Add dilation to average pooling with `AvgPoolOptions::with_dilation`, matching max pooling.
  CubeCL backends use CubeK kernels for dilated 1D/2D average pooling, preserving the NHWC layout
  optimization. `count_include_pad` counts the dilated taps that land in padding. **Breaking:** Add
  `dilation: [1; N]` to `AvgPoolOptions` struct literals. **Breaking for third-party backends:**
  The `ModuleOps` average pooling methods and their backward passes take a `dilation` argument after
  `padding`, as do `module::avg_pool2d_backward` and `module::avg_pool3d_backward`.
  ([#6235](https://github.com/tracel-ai/burn/pull/6235))
  [@ashokram-si](https://github.com/ashokram-si)

- Fix CubeCL `max_pool3d`, which did not compile on WebGPU or any other WGSL target.
  [@vboussot](https://github.com/vboussot)

- Correct max-pooling SAME padding with dilation.
  ([#6023](https://github.com/tracel-ai/burn/pull/6023))

- Fix incorrect `max_pool3d` gradients on CubeCL backends.
  ([#6428](https://github.com/tracel-ai/burn/pull/6428)) [@laggui](https://github.com/laggui)

#### Stateful feed-forward activations

Stateful `PositionWiseFeedForward` activations such as PReLU and SwiGLU are now trained and
persisted. Older checkpoints lack these activation parameters; retain their initialized values when
loading those checkpoints by filtering to `linear_inner` and `linear_outer`.
([#5932](https://github.com/tracel-ai/burn/pull/5932))

### Training and data

#### Binary cross-entropy loss

- Reject invalid binary-cross-entropy inputs.
  ([#5969](https://github.com/tracel-ai/burn/pull/5969))

#### Classification metrics

- Reduce classification metric memory use by storing confusion codes as U8 on devices that support
  it, retaining the default integer dtype otherwise.
  ([#6224](https://github.com/tracel-ai/burn/pull/6224))
  [@Mikyx-1](https://github.com/Mikyx-1)

- Add the Matthews correlation coefficient metric for binary and multiclass classification.
  ([#5818](https://github.com/tracel-ai/burn/pull/5818))

- Default the F-beta metric to F1 and require a finite, positive beta.
  ([#5965](https://github.com/tracel-ai/burn/pull/5965))

#### CTC loss

- Improve half-precision CTC loss accuracy. Fix CTC execution through router and remote backends
  and reject CTC inputs exceeding the device shared-memory limit.
  ([#6103](https://github.com/tracel-ai/burn/pull/6103),
  [#6001](https://github.com/tracel-ai/burn/pull/6001),
  [#6029](https://github.com/tracel-ai/burn/pull/6029))

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

- Add `DataLoaderBuilder::drop_last(bool)` and `FixBatchStrategy::with_drop_last(bool)` to discard
  the final incomplete batch, including with multiple workers. The default keeps it. Expected
  output counts and progress exclude dropped samples; `num_items()` keeps reporting the input
  dataset size. ([#5911](https://github.com/tracel-ai/burn/pull/5911))
  [@ProgrammerProfi](https://github.com/ProgrammerProfi)

- Flush partial gradient accumulation only after normal dataloader exhaustion. An interruption or
  error observed before that update discards the pending window.
  ([#6184](https://github.com/tracel-ai/burn/pull/6184))

#### Datasets

- Return errors instead of panicking when `InMemDataset::from_json_rows` encounters unreadable
  lines, invalid UTF-8, or JSON that cannot be deserialized.
  ([#6284](https://github.com/tracel-ai/burn/pull/6284))
  [@onenewcode](https://github.com/onenewcode)

- Add `ImageFolderDataset::class_name` to look up the class name of a label.
  ([#6310](https://github.com/tracel-ai/burn/pull/6310))

#### LBFGS optimizer

- Keep LBFGS parameter updates aligned when some parameters have no gradients, including when
  optimizer history exists. ([#6121](https://github.com/tracel-ai/burn/pull/6121))

#### LPIPS pretrained weights

- Reject incomplete LPIPS pretrained weights instead of silently using random parameters, and
  correct the AlexNet checkpoint. Use `try_init_pretrained` or `try_load_pretrained_weights` to
  handle loading failures without panicking. ([#6067](https://github.com/tracel-ai/burn/pull/6067))

#### Optimizer-step metrics

**Breaking:** Replace `metric_train_numeric(LearningRateMetric::new())` with
`metrics(LearningRateMetric::new())`. Learning rates are recorded per applied optimizer update
and parameter group, including partial accumulation windows. Custom strategies must emit
`LearnerEvent::OptimizerStep` after each update.
Use `OnOptimizerStep::with(...)` to register standard metrics for optimizer updates.
See [optimizer-step metrics](./building-blocks/metric.md#optimizer-step-metrics).

#### Optimizers

- Build `burn-optim` without the `std` feature again. `ModuleOptimizer` no longer depends on the
  standard-library-only learning-rate scheduler module, so fixed learning rates work in `no_std`.
  ([#6475](https://github.com/tracel-ai/burn/issues/6475))

#### Regression metrics

- Add regression evaluation metrics (`MseMetric`, `MaeMetric`, `R2ScoreMetric`) to
  `burn-train`, sharing `RegressionInput` and supported directly by `RegressionOutput`.
  R² accumulates centered statistics across batches and averages target scores uniformly.
  ([#6390](https://github.com/tracel-ai/burn/pull/6390),
  [#6042](https://github.com/tracel-ai/burn/pull/6042))
  [@divakantpandey](https://github.com/divakantpandey), [@Infinty-ux](https://github.com/Infinty-ux)

#### Text generation metrics

- Correct BLEU and ROUGE-L scores and prevent panics when prediction and reference sequence lengths
  differ. ([#6242](https://github.com/tracel-ai/burn/pull/6242))
  [@onenewcode](https://github.com/onenewcode)

- Fix `BleuScore` running and final values to use corpus-wide statistics instead of averaging batch
  scores. ([#6240](https://github.com/tracel-ai/burn/pull/6240))
  [@Arc000007](https://github.com/Arc000007)

#### Text recognition metrics

- Correct CER padding handling and WER calculations when reference and prediction sequence
  lengths differ.
  ([#5945](https://github.com/tracel-ai/burn/pull/5945),
  [#6028](https://github.com/tracel-ai/burn/pull/6028))

### Checkpoints

#### Burnpack reader limits

- Add configurable `burn_pack::ReaderLimits` through `Reader::with_limits`.
  ([#6007](https://github.com/tracel-ai/burn/pull/6007))

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

- Speed up loading strided and transposed PyTorch checkpoint weights.
  ([#6041](https://github.com/tracel-ai/burn/pull/6041),
  [#6113](https://github.com/tracel-ai/burn/pull/6113))

#### PyTorch checkpoint loading

- Add `PytorchReader::from_bytes` for loading checkpoints from memory.
  ([#5755](https://github.com/tracel-ai/burn/pull/5755))

#### SafeTensors loading

- Avoid repeated SafeTensors header parsing when loading models with many tensors.
  ([#6269](https://github.com/tracel-ai/burn/pull/6269))
  [@Arc000007](https://github.com/Arc000007)

### Extensions

#### Linear algebra (`burn-linalg`)

##### Batched linear solves

`burn::linalg::solve` solves square, nonsingular systems `A @ X = B` with vector or matrix
right-hand sides and broadcast batches. Flex and CubeCL provide optimized factorization and
substitution paths with autodiff support. F16 and BF16 inputs compute in F32 and cast back;
nonsquare or singular systems are rejected.
([#5856](https://github.com/tracel-ai/burn/pull/5856))
[@Mikyx-1](https://github.com/Mikyx-1)

##### Cholesky decomposition

`burn::linalg::cholesky(a, upper)` factors real symmetric positive-definite matrices, including
batches, with optimized Flex CPU and supported CubeCL GPU paths. Autodiff uses the portable tensor
implementation. Pass `false` for a lower triangular factor or `true` for an upper triangular factor;
only the selected input triangle is read. Non-positive or non-finite pivots are rejected.
([#5858](https://github.com/tracel-ai/burn/pull/5858))
[@Mikyx-1](https://github.com/Mikyx-1) [@laggui](https://github.com/laggui)

#### Signal processing (`burn-signal`)

##### FFTs of arbitrary sizes

Real FFTs and their inverses now accept arbitrary sizes, removing the power-of-two restriction.
Complex inverse FFT (`ifft`) is also available.
([#5840](https://github.com/tracel-ai/burn/pull/5840),
[#5757](https://github.com/tracel-ai/burn/pull/5757))

#### Vision (`burn-vision`)

##### Connected components

- Restore GPU acceleration for connected components.
  ([#6003](https://github.com/tracel-ai/burn/pull/6003))

##### Median blur

- Speed up Flex CPU median blur for F16, BF16, F32, F64, and integer tensors by selecting
  neighborhood medians without stacking the images. U8, I8, and exact normalized RGB8 inputs
  use sliding histograms for larger filters. Floating-point autodiff uses the selected source
  pixels for backward. `median` and `MedianBlur` now also accept `Tensor<4, Int>` images;
  specify `Tensor::<4>` on float constructors if their kind was previously inferred from these calls.
  ([#6219](https://github.com/tracel-ai/burn/pull/6219))
  [@Mikyx-1](https://github.com/Mikyx-1)

##### Non-maximum suppression

- Fix CPU non-maximum-suppression results.
  ([#6019](https://github.com/tracel-ai/burn/pull/6019))
