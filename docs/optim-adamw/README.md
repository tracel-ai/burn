# AdamW execution optimization

This change adds opt-in execution strategies for the existing AdamW algorithm. The delivery is
one Burn PR covering single-parameter fusion, safe output reuse, measured kernel throughput, and
module-level batching where its dependency and performance requirements can be satisfied.
The default algorithm, configuration defaults, parameter groups, and checkpoint format remain
the compatibility contract.

The branch is still under development. Single-parameter fusion, copy-on-write output reuse, and
CPU vectorization are present. Foreach is not yet integrated: the pinned CubeCL revision
`3e6d4f35c26e486728d4f46f00d0cf84f47b2171` cannot publicly report a client's effective execution
stream. New Linux CPU/CUDA validation is in progress. Historical measurements and review logs
precede the upstream merge and do not establish that the current revision passes or improves
performance.

## Execution and ownership

`AdamWConfig::with_execution(Some(AdamWExecutionConfig::Fused))` selects the dedicated path.
An absent execution field, including one in an older serialized config, selects `Reference`.
The non-exhaustive execution enum will expose Foreach only with a working implementation.

The responsibilities are divided at the existing boundaries:

| Layer | Responsibility |
| --- | --- |
| `ModuleOptimizer` | Select parameter groups and learning rates, skip missing gradients, apply per-parameter clipping, migrate state, and restore parameter context. |
| `burn-optim` AdamW bridge | Convert eligible tensors and state to raw primitives without losing fallback inputs, and reconstruct the same typed state afterwards. |
| `burn-cubecl` AdamW kernel | Validate layouts and launch limits, choose each output's storage, and execute the numeric update. |
| CubeCL runtime | Own allocations, streams, compilation, and kernel submission. |

The dedicated kernel belongs in the shared CubeCL backend, rather than separately in CUDA,
Metal, WGPU, and CPU runtime selectors. It is reached through an optional `burn-cubecl`
dependency, without adding an optimizer-specific operation to every `Backend` implementation.
`burn-optim` runtime features are forwarded by the public `burn` crate.

Raw F32 tensors can use the specialized path. Fusion tensors represent deferred operations;
using their buffers directly would bypass the graph. Fusion, unsupported dtypes/layouts, and
failed primitive conversions therefore use the tensor-operation reference. A direct step on a
tracked autodiff tensor also falls back; the module optimizer removes and restores that context.
This limits where the optimization helps: enabling a runtime with default fusion features does
not automatically enable a faster optimizer. Both configurations require validation.

Fusion removes the intermediate tensors between the first-moment, second-moment, bias,
decay, and parameter-update operations. An initialized update without AMSGrad logically reads
four F32 streams and writes three: 28 bytes per element, before metadata, cache effects, and
allocator traffic. This is a model of the algorithm's traffic, not measured bandwidth or a
claim about the reference's physical launch count.

Output reuse is a separate decision for the parameter, first moment, second moment, and
AMSGrad maximum. Reuse requires exclusive ownership, a compatible contiguous layout, and
no allocation alias with another input role. Shared tensors remain unchanged; aliases use
fresh outputs. All primitive conversions must succeed before temporary wrapper clones are
released and ownership is checked. Rejection reconstructs the complete original state.
The batch path must use the same ownership rules, including aliases across batch items.

CPU vectorization addresses throughput within a parameter. Every participating input and
output must support the chosen width and alignment; a partial vector uses a scalar tail.
GPU tuning needs its own evidence. A width that improves the CPU is not a justified default
for CUDA, Metal, or WGPU.

## Numeric and state contract

The reference remains the numerical anchor, with `beta_1=0.9`, `beta_2=0.999`,
`epsilon=1e-5`, `weight_decay=1e-4`, and AMSGrad/cautious decay disabled by default.
Every implementation must preserve:

- Host scalar precision and evaluation order: f32 beta factors and bias divisors,
  `state.time as i32` for `powi`, and f64 learning-rate/decay products until conversion to the
  input dtype. Bias division must not silently become multiplication by a reciprocal.
- Moment updates from the clipped gradient. First-step moments are initialized without
  reading old state. Missing gradients do not advance a parameter's time.
- The no-decay path without introducing `0 * Inf`. Cautious decay compares the parameter's
  and updated first moment's signs separately; multiplying them changes underflow and
  signed-zero behavior.
- AMSGrad's `old_max < new_moment` selection: ties and an old NaN keep the old value; a new
  NaN does not replace a finite old maximum. Enabling AMSGrad without stored history starts
  from the new second moment. Disabling it preserves existing history without using or
  updating it.
- Shape, rank, dtype, device, time, checkpoint names, and the persisted `__rank`. Existing
  dynamic rank limits must fail at the same boundary. Checkpoints must resume across
  execution modes and AMSGrad transitions.
- Shared-input values, parameter mapping, gradient-checkpointing strategy, and distributed
  context. Clipping remains per parameter; its coefficient cannot be folded into learning
  rate because it affects the moments before normalization.

F32 results use justified dtype-based tolerances; exceptional values need classification or
bit-pattern assertions. The implementation preserves separate multiplication rounding where
compiler FMA contraction would otherwise change behavior. F16/BF16 support cannot be inferred
from F32 tests: a fused higher-precision chain can avoid an overflow that the reference
materializes between operations. No FP32 master weights or checkpoint conversion is added.

## Module batching requirements

Foreach targets launch and host-dispatch costs across independent parameters. It does not
concatenate the model, change its layout, or reduce the per-element traffic of an already fused
update. A viable implementation must account for all collection, grouping, output preparation,
launch, and writeback costs.

The current module mapper returns each parameter immediately. Deferred batching therefore
needs collection followed by execution and writeback. The retained parameter handles can
prevent parameter-buffer reuse; state buffers can still be exclusive. The benchmark must
capture this cost, and inputs that cannot join a batch should take the existing single-step
path before collecting extra handles. Batch eligibility must not require a second traversal
for modules that have no eligible updates.

Compatible items must agree on device, client service, effective execution stream, dtype,
learning rate, step count, beta/epsilon/decay values, and the state/AMSGrad/decay variants.
Each tensor retains its own shape and strides. Batches must respect argument bindings,
workgroup and grid limits, and bounded descriptor specialization. Empty groups do nothing;
singletons and oversized items use the existing fused/reference fallback. No failure after
submission may retry an update and apply it twice.

The stream is an execution property, not the allocation's creation stream. A pinned client
may differ from the ambient stream. Collection must capture the effective stream without
synchronization or side effects; allocations, delayed singletons, launches, and fallback must
execute under the same context and restore the caller's ambient stream. A public read-only
CubeCL getter, a consistent Burn/CubeK dependency graph, and a normal downstream build are
prerequisites. A local Cargo patch alone is not a shippable dependency solution.

## Comparison with mainstream implementations

The comparison below uses versioned framework source and JAX's buffer-donation documentation,
not performance measurements between frameworks.

**PyTorch 2.9.0** provides three execution paths in its shared Adam/AdamW implementation.
The single-tensor path applies operations parameter by parameter. The foreach path groups
lists by device and dtype and applies each arithmetic operation across a list: horizontal
fusion reduces launch overhead but still leaves a sequence of optimizer operations. Its fused
path calls `_fused_adamw_` for each compatible group and combines the update itself. Burn's
single-parameter `Fused` is vertical fusion; the planned Foreach combines that update across
parameters and is consequently closer to PyTorch's fused optimizer than to its foreach-only
arithmetic path.

PyTorch's CUDA multi-tensor implementation maps blocks to tensor/chunk descriptors instead of
padding every parameter to the largest shape. In the cited version it uses 65,536-element
chunks, 512-thread blocks, and four-element instruction-level parallelism. Tensor and block
counts are bounded by metadata/argument capacity. Burn should adopt the principle of bounded
work descriptors, not copy these CUDA constants into CPU or WebGPU. CubeCL binding limits,
metadata layout, vector widths, and compiler specialization costs differ by runtime.
PyTorch also carries per-tensor step counters in its fused metadata. Burn's proposed shared
scalar bundle requires equal learning rate and step count within a group. That keeps kernel
arguments simple but can split a model with missing gradients or parameter-specific schedules
into more launches; per-item scalar metadata is justified only if measurements outweigh its
extra preparation and argument cost.

PyTorch updates optimizer-owned moments and parameter buffers in place. Burn's tensor values
may share storage with externally retained handles, so unconditional mutation would violate
its API. Per-output COW preserves that contract and makes exclusive inputs cheap. Conversely,
the two-pass module collection can retain parameters and remove that advantage; fewer launches
alone do not prove a faster step.

The arithmetic is also not interchangeable. PyTorch's single-tensor and foreach paths use
`lerp_` for the first moment and `addcmul_` for the second. Its specialized CUDA kernel uses
an accumulator type and combines learning rate and bias correction differently from Burn.
Those choices may change rounding and overflow relative to Burn's tensor-operation sequence.
Burn must preserve its own contract rather than use PyTorch parity as a reason to relax it.
PyTorch also supports execution options and dtypes beyond this change's raw F32 scope.

**Optax 0.2.6 with JAX** expresses AdamW as a chain of `scale_by_adam`,
`add_decayed_weights`, and `scale_by_learning_rate` over a parameter tree. The functional
interface produces updates and new state. With a compiled training step, the compiler can
optimize that graph; optional JAX buffer donation lets it reuse inputs the caller relinquishes.
This differs from Burn's eager specialized-kernel dispatch and dynamic COW checks. The useful
architectural lesson is to keep optimizer semantics separate from execution and storage
decisions. Optax's composition alone does not guarantee a particular launch count or storage
reuse, and JAX donation requires an explicit lifetime promise from its caller.

Sources:

- [PyTorch Adam/AdamW execution paths](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/adam.py)
- [PyTorch CUDA multi-tensor descriptors](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/MultiTensorApply.cuh)
- [PyTorch fused CUDA AdamW arithmetic](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/fused_adam_utils.cuh)
- [Optax AdamW composition](https://github.com/google-deepmind/optax/blob/v0.2.6/optax/_src/alias.py#L599-L728)
- [JAX buffer donation](https://github.com/jax-ml/jax/blob/a038d44f66d2f79804d285896ac5b8c4e032a3ed/docs/buffer_donation.md)

## Evidence required before the PR is ready

The original optimization scope remains open until the following are demonstrated on the
final source. Removing obsolete experiment files does not remove these requirements.

| Requirement | Evidence needed |
| --- | --- |
| Single-parameter fusion | Actual specialized launches and parameter/all-state comparisons against Reference over multiple steps, decay variants, layouts, and special values. |
| COW reuse | Independent reuse and alias tests for all outputs; direct comparison with out-of-place fusion; allocation and peak-memory results, including shared inputs. |
| Throughput | Scalar-versus-vector comparison at 1/4/16 large parameters, tails, unaligned offsets, and small controls; variant/JIT and code-size costs. |
| Foreach | Real module batch launches, per-parameter learning rates/time, missing gradients, groups, clipping, mixed rank/layout, chunk limits, streams, aliases, checkpoint transitions, and direct comparison with the best single-parameter path. |
| Fallbacks | Default Flex, raw and fusion builds, tracked tensors, unsupported dtypes/layouts, and actual capture/router/shard/remote behavior where applicable; skipped cases identified explicitly. |
| Integration | Consistent published dependencies, downstream consumption without a root patch, feature/no-std checks, book/changelog updates, and `cargo run-checks`. |
| Review and delivery | Two reviews of each final step before its commit, findings resolved and rechecked, and one PR describing the final scope with accurate AI disclosure. |

Current Linux validation uses an Intel Core i9-14900HX and NVIDIA RTX 5070 series GPU. No
current performance result is claimed yet. Hardware, driver, OS, Rust/Cargo versions, backend,
features, build profile, source hashes, and binary hash must accompany accepted results.

Use the existing `burn-optim` AdamW benchmark on a raw runtime build:

```sh
RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= cargo bench -p burn-optim --bench adamw \
  --no-default-features --features std,cuda --no-run --locked
BURN_DEVICE=cuda <built-binary> --verify-path
BURN_DEVICE=cuda <built-binary> --samples-json reference tiny
BURN_DEVICE=cuda <built-binary> --samples-json fused tiny
```

Measure compilation separately from steady state. Compare initialized independent inputs,
warm both modes, alternate execution order, synchronize at equivalent boundaries, and retain
raw samples. Include enqueue and full step-plus-sync median/p95, allocation/live/peak memory,
first-step latency, and actual launch counts. For incremental claims, change only the feature
being evaluated; cumulative Reference-to-final gains cannot establish each stage's benefit.
Run benchmarks serially on an otherwise idle machine, without concurrent builds. A faster
kernel is insufficient if module traversal, metadata, allocation, or synchronization makes the
whole step slower. Report regressions and restrict runtime eligibility when the evidence does
not support enabling a path.

The adjacent measurement scripts retain their separate purposes: `measure.py` records a
single binary's runs, `compare.py` alternates two binaries with matching source manifests, and
`summarize.py` summarizes recorded observations. A source manifest must be captured with the
exact built binary; hashing the current checkout after a different build is not provenance.
Historical drafts, patch archives, and superseded measurements are recoverable from git
history; they are not part of the current result set.
