# AdamW execution optimization

This change adds opt-in execution strategies for the existing AdamW algorithm. The delivery is
one Burn PR covering single-parameter fusion, safe output reuse, measured kernel throughput, and
module-level batching where its dependency and performance requirements can be satisfied.
The default algorithm, configuration defaults, parameter groups, and checkpoint format remain
the compatibility contract.

Single-parameter fusion, copy-on-write output reuse, CPU vectorization, and CPU module-level
Foreach are implemented. The execution-stream prerequisite is published and pinned, and
singleton launches wait until the module releases its retained parameter. Current Linux
measurements show benefits for specific workloads and regressions for others; Foreach stays
opt-in. The recorded snapshot passed focused checks, complete repository validation and
published downstream consumption; its measurements show the large-batch gain and the unaligned
scalar fallback's limits. Those results precede the 2026-11-08 upstream synchronization and
retain their original source identities. Earlier superseded experiments are excluded.

On 2026-11-08 the branch synchronized with upstream `main` at
`43acf2ded51bf7be46c71666b7d5705ec21e7a8e` and refreshed CubeCL/CubeK to the revisions below.
The sync also required a feature-isolation fix: the new
upstream collective `run_member` method and `GroupMember` import moved unchanged into the
existing fusion-only module after strict clippy found them unused without fusion. No warning
suppression was added, and the AdamW implementation is unchanged. Focused and full repository
validation passed on the new pins; a rebuilt CUDA comparison is recorded separately below.

## Execution and ownership

`AdamWConfig::with_execution(Some(AdamWExecutionConfig::Fused))` selects the dedicated
single-parameter path. `Foreach` requests batching of compatible small CPU module parameters.
Direct calls, other runtimes, large inputs, and singleton groups use `Fused`, including its
reference fallback. An absent execution field, including one in an older serialized config,
selects `Reference`. The execution enum is non-exhaustive.

The responsibilities are divided at the existing boundaries:

| Layer | Responsibility |
| --- | --- |
| `ModuleOptimizer` | Select groups/rates, skip missing gradients, clip and migrate state, collect eligible updates, and restore parameter/state/context using the module's original ranks. |
| `burn-optim` AdamW bridge | Convert eligible tensors and state to raw primitives without losing fallback inputs, share conversion between immediate and deferred execution, and reconstruct typed state. |
| `burn-cubecl` AdamW kernel | Validate layouts and budgets, group complete execution contexts, choose each output's storage, and execute the numeric update or return an unlaunched singleton. |
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
The batch path uses the same ownership rules, including aliases across batch items.

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

## Module batching and deferred singletons

Foreach targets launch and host-dispatch costs across independent parameters. It does not
concatenate the model, change its layout, or reduce the per-element traffic of an already fused
update. A viable implementation must account for all collection, grouping, output preparation,
launch, and writeback costs.

The first module traversal handles group selection, clipping, state migration, and scalar
selection before attempting preparation. Preparation neither allocates outputs nor launches
kernels. Unsupported or oversized items take the existing single-step path immediately. If
no item is collected, there is no second traversal. Otherwise the backend groups pending
items and returns results in input order; a second module traversal provides each original
rank for reconstruction without boxed state builders or a new dynamic-rank dispatcher.

For a real multi-tensor batch, execution occurs before writeback while the module still holds
its original parameters. This deliberately preserves module semantics but prevents those
parameter buffers from being exclusively owned. Independently owned moment buffers can still
be reused. The extra parameter allocation is part of the end-to-end cost that the benchmark
must measure.

An item that cannot find a compatible partner now returns `AdamWBatchResult::Deferred`, while
a submitted batch returns `Updated`. During writeback, the mapper consumes the old `Param`,
drops its retained tensor, and only then calls `finish()`. The deferred item runs the existing
single-parameter kernel, which can recover exclusive parameter ownership and its usual COW
reuse. An `Updated` result simply returns its completed output. This distinction records
whether work has actually been submitted and prevents retrying an already applied update.
It does not remove the collection/writeback cost for fragmented singleton groups. Group
lookup remains a linear scan; its cost must be measured separately from the ownership fix.

Compatible items must agree on device, client service, effective execution stream, dtype,
learning rate, step count, beta/epsilon/decay values, and the state/AMSGrad/decay variants.
Floating-point scalars are compared by bit pattern. Each tensor retains its own shape and
strides. The current CPU implementation uses 4,096-element chunks and at most eight tensors
and 32 chunks per launch, reduced further by runtime binding/grid limits. An individual item
occupying the whole chunk budget is refused before collection because it cannot share a
batch. Chunk descriptors bound specialization and avoid padding each tensor to the largest
shape. These constants are CPU policy, not a GPU performance claim.

The stream is an execution property, not the allocation's creation stream. A pinned client
may differ from the ambient stream. Collection must capture the effective stream without
synchronization or side effects; allocations, delayed singletons, launches, and fallback must
execute under the same context and restore the caller's ambient stream. A public read-only
CubeCL getter, a consistent Burn/CubeK dependency graph, and a normal downstream build are
prerequisites. A local Cargo patch alone is not a shippable dependency solution.

The current normal Git dependency graph is:

| Dependency | Published revision | Change |
| --- | --- | --- |
| [CubeCL](https://github.com/onenewcode/cubecl/commit/5a9e57489357e0336dae0ff23df502b1896be397) | `5a9e57489357e0336dae0ff23df502b1896be397` | Preserve the read-only `Client::execution_stream()` getter and CPU scope/pinning tests on the updated upstream. |
| [CubeK](https://github.com/onenewcode/cubek/commit/f67dce4244d89fff2c809bd5c3194620612bf682) | `f67dce4244d89fff2c809bd5c3194620612bf682` | Merge updated upstream and pin CubeCL dependencies to the same getter revision. |

CubeCL's change is based on upstream `cb1b7bc6e521b2556ed01cf096b99104a50b35ae`;
CubeK's dependency-only change is based on `24ea8e9f2c3baff39fcb1db5efef1872c00e7b86`.
Burn and CubeK resolve CubeCL workspace packages to one Git source. `cubecl-hip-sys` is a
separate native binding from crates.io.

The earlier standalone downstream consumer resolved 22 CubeCL and 13 CubeK packages using
CubeCL `e97ef2c4dc06e18281effe0b71e1d361b2d1e4c4` and CubeK
`2fd57bac2d09c84ab3da08d5bacb6b737646182b`, then executed public Foreach APIs against
published Burn commit `639ad952b838d39e43f20db7717d914ef9b58a47`. This is evidence for those
older pins, not the updated dependency graph. Its 652 dependencies were
resolved afresh; `cargo run --locked` then passed. The fixture and its source/log hashes are
embedded in [results.json](results.json). Reproduction resolves transitive dependencies again
before the locked run, so later runs may choose newer compatible versions. The verifier rejects
local path dependencies and root manifest patches; it does not independently audit Cargo source
replacement.

## Comparison with mainstream implementations

The comparison below uses versioned framework source and JAX's buffer-donation documentation,
not performance measurements between frameworks.

**PyTorch 2.9.0** provides three execution paths in its shared Adam/AdamW implementation.
The single-tensor path applies operations parameter by parameter. The foreach path groups
lists by device and dtype and applies each arithmetic operation across a list: horizontal
fusion reduces launch overhead but still leaves a sequence of optimizer operations. Its fused
path calls `_fused_adamw_` for each compatible group and combines the update itself. Burn's
single-parameter `Fused` is vertical fusion; Foreach combines that update across
parameters and is consequently closer to PyTorch's fused optimizer than to its foreach-only
arithmetic path.

PyTorch's CUDA multi-tensor implementation maps blocks to tensor/chunk descriptors instead of
padding every parameter to the largest shape. In the cited version it uses 65,536-element
chunks, 512-thread blocks, and four-element instruction-level parallelism. Tensor and block
counts are bounded by metadata/argument capacity. Burn should adopt the principle of bounded
work descriptors, not copy these CUDA constants into CPU or WebGPU. CubeCL binding limits,
metadata layout, vector widths, and compiler specialization costs differ by runtime.
PyTorch also carries per-tensor step counters in its fused metadata. Burn's shared
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

## Validation status

The `upstream_sync` section of [results.json](results.json) records the 2026-11-08 snapshot;
the original evidence remains intact with its earlier CubeCL `e97ef2c4`/CubeK `2fd57bac` pins.
New full logs and commands are under `target/adamw-evidence/upstream-sync-2026-11-08`.

| New-pin check | Observed result and boundary |
| --- | --- |
| Raw CPU and CUDA AdamW | Each reported 50 passes, 0 failures and 2 ignored tests; both isolated CPU JIT tests passed individually. |
| Decorator integration | All 3 tests passed: split shards, loopback remote and Capture IR. |
| CPU and CUDA fusion AdamW | Each passed again after the collective feature-isolation fix: 50 passes, 0 failures and 2 ignored tests. Raw-only cases return early; these are not 50 specialized-kernel executions. |
| Host fusion unit suite | `burn-fusion --lib`: 144 passed, 1 ignored throughput timing harness; includes stream ordering for skipped operations and error paths. Native multi-GPU collective execution was not tested. |
| Lint and repository checks | Strict all-target clippy passed with `std,cpu,cuda,burn-cubecl/autotune`; complete `cargo run-checks` passed, with 29 test summaries totaling 4,659 passes, 0 failures and 39 ignored. The initial clippy failure and read-only cache-artifact failure remain recorded; private writable artifact copies enabled the repository retry. |
| Rebuilt executable | All 9 CPU path/value checks passed; CUDA memcheck covered the same 9 workloads and reported zero errors. |

The rebuilt binary is `6f8359ec2772236703287b75fdd7520313d080f6267302472af50835035cf102`.
Its 184 covered source hashes match the validated tree; `upstream_sync.merge_proof` also records
upstream changes outside that build subset. CUDA `large_16` was measured in three processes
per mode, 30 samples each, with alternating process order:

| New-pin CUDA comparison | Reference median / p95 ms | Fused median / p95 ms | Median ratio | Per-process ratios |
| --- | ---: | ---: | ---: | --- |
| `large_16` | 1.4549 / 1.4870 | 1.0744 / 1.0963 | 1.354× | 1.357, 1.357, 1.354 |

This comparison measures Reference versus single-parameter Fused execution. Differences from
the earlier series cannot establish a dependency-induced performance change. The broader CPU,
COW, vector and Foreach matrix and standalone downstream consumer were not repeated on these
pins; their earlier results remain attributed to their original snapshots.

The following detailed table describes that earlier snapshot:

| Check | Observed result and boundary |
| --- | --- |
| Raw CPU AdamW unit suite | Corrected-source rerun: 50 reported passes, 2 intentionally ignored process-wide JIT-key tests. |
| Raw CUDA AdamW unit suite | Recorded rerun: 50 reported passes, 2 ignored JIT-key tests. The count includes CPU-only cases that return on CUDA; it is not 50 independent GPU executions. |
| Isolated CPU JIT tests | Both ignored tests were run individually and passed: vector width keys and Foreach descriptor keys. |
| Decorator integration target | 3 passed, none ignored: real split shards, actual loopback WebSocket CPU execution, and Capture IR equivalence. Shard/remote exercise direct and module paths, two steps, all AMSGrad state, serialized state recovery, and parameter context; Capture currently exercises direct steps and all eight output IDs. |
| CubeCL getter | 2 CPU integration tests passed, focused clippy passed, runtime `--no-default-features` check passed. Full CubeCL workspace validation was not run. |
| CubeK dependency graph | Normal and locked/offline metadata checks passed without root patches; only the intended CubeCL source replacements changed in its lockfile. |
| Flex optimizer suite | 140 passed, none ignored. |
| CPU fusion build | Corrected-source rerun: 50 reported passes, 2 ignored. The initial F16 assertion failure is retained in the history; the corrected test compares independently constructed reference execution and retains the overflow assertion only for eager F16. Raw-only tests return early, so the pass count is not a specialized-kernel execution count. |
| No-std | `cargo check -p burn-optim -p burn --locked --no-default-features` passed without warnings after explicit allocation imports and no-std scheduler fixes. Initial missing-import failures remain in the history. |
| Scheduler regression suite | 49 passed with Flex after exposing the existing scheduler module to no-std builds. |
| Focused clippy | All targets passed with `std,cpu,cuda,burn-cubecl/autotune` and `-D warnings` after the recorded lint fixes. This result applies to that feature set. |
| Repository checks | `cargo run-checks` passed: formatting, typos, dependency audit, workspace clippy, no-std, feature isolation and backend tests; 29 test executables reported 4,659 passes, 0 failures and 39 ignored. Non-writable cache artifacts blocked earlier attempts; pass-through Rust wrappers and writable build artifacts enabled the completed run. The raw evidence JSON alone is excluded from typos because it contains generated kernel identifiers. Failed attempts remain in the history. |
| Downstream consumer | Passed against published Burn `639ad952b838d39e43f20db7717d914ef9b58a47` using normal Git dependencies: two independent two-parameter models, two steps, parameter/all-six-state-tensor comparisons, serialized scalars/paths, and exactly two real CPU batch launches. |
| Reviewed benchmark | CPU candidate/scalar and CUDA candidate path/value checks passed. CUDA `compute-sanitizer --tool memcheck` reported zero errors over the nine verification workloads, including the offset input. Benchmark clippy and formatting passed after that fixture was added. |

Reproduce the raw unit suite with `BURN_DEVICE=cpu`, then `BURN_DEVICE=cuda`:

```sh
BURN_DEVICE=cpu RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  cargo test -p burn-optim --locked --no-default-features --features std,cpu,cuda \
  optim::adamw:: -- --test-threads=1
BURN_DEVICE=cpu RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  cargo test -p burn-optim --locked --no-default-features --features std,cpu,cuda \
  vector_width_has_distinct_and_reusable_jit_keys -- --ignored --test-threads=1
BURN_DEVICE=cpu RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  cargo test -p burn-optim --locked --no-default-features --features std,cpu,cuda \
  foreach_jit_reuses_stable_layout_and_distinguishes_descriptors -- --ignored --test-threads=1
BURN_DEVICE=cpu RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  cargo test -p burn-optim --locked --no-default-features \
  --features std,cpu,cuda,adamw-execution-tests --test adamw_execution -- --test-threads=1
```

The raw backend builds emit convolution/utility warnings; the integration test itself compiled
without warnings. Metal/WGPU/ROCm have no current hardware results in this evidence set.
Capture coverage is IR fallback equivalence, not a claim that CUDA hardware graph replay can
advance host learning rates or step counters.

## Measured results and costs

The measurements in this section and the following confirmation table use the earlier pins
`e97ef2c4`/`2fd57bac`; they are not reattributed to the 2026-11-08 dependency update.
The [durable evidence file](results.json) retains raw numeric samples, process/repeat identity,
source-report hashes, environment, launch assertions, exact ablation patches, and build
manifests deduplicated by binary hash. Each manifest's full source map is reconstructed by
merging `common_sources` with its `sources`. All ratios below are baseline time divided by
candidate time; values below 1 mean the candidate is slower. They are measurements on one
machine, not cross-framework or cross-hardware claims.

The measured direct candidate is `47483572bf4a2fa6468faf893e731beb61e9742a5989f022f26f17790e8bddf8`.
Module paired runs use `ed1710cde90ae4d680bec37578351bc3527f3234a19bab39bb1a70544f382c04`;
its only source difference is a comment beside vector-width selection, verified by kernel
hashes and preserved as an exact patch in the evidence. Later validation edits are not
silently attributed to either measured binary.

Direct Reference-to-candidate measurements include all implemented single-parameter improvements.
Three processes per mode produced 30 steady samples each, alternating process order:

| Workload | CPU Reference / Fused median ms | Ratio | CUDA Reference / Fused median ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `tiny` | 73.3897 / 1.0002 | 73.372× | 1.3328 / 0.2370 | 5.624× |
| `mixed_rank` | 3.2670 / 0.1192 | 27.415× | 0.1688 / 0.0393 | 4.295× |
| `large_1` | 0.9255 / 0.1295 | 7.147× | 0.1096 / 0.0274 | 3.997× |
| `large_4` | 12.0721 / 1.6186 | 7.458× | 0.3740 / 0.2254 | 1.659× |
| `large_16` | 117.7953 / 7.0661 | 16.670× | 1.3392 / 0.8778 | 1.526× |
| `long_short` | 143.6463 / 1.3982 | 102.736× | 1.5264 / 0.2810 | 5.432× |

CPU small-workload timings vary substantially between processes: `tiny` ratios range from
45.871× to 157.882× and `mixed_rank` from 4.126× to 23.513×. The pooled ratios are descriptive,
not stable predictions. These cumulative comparisons cannot assign the benefit to COW,
vectorization, or fusion independently.

The incremental ablations each change one policy in that measured executable's source: disable
single-step output reuse, or force its vector width to one. Four alternating before/after
process pairs each contain 30 samples. Shared-input controls retain every old parameter and
state handle outside the timed step.

| Incremental comparison | Workload | Pooled ratio | Per-process-pair ratio range / interpretation |
| --- | --- | ---: | --- |
| CPU out-of-place → COW | `large_4` | 1.428× | 1.388–1.483× |
| CPU out-of-place → COW | `large_16` | 1.697× | 1.684–1.762× |
| CPU out-of-place → COW | `long_short` | 9.230× | 7.191–11.317×; allocator-sensitive |
| CPU out-of-place → COW | `mixed_rank`, `large_1` | 1.488×, 1.223× | Each has one pair below 1; no consistent per-run gain |
| CPU COW with shared inputs | `tiny`, `large_16` | 1.116×, 0.957× | Reuse is forbidden; no reliable improvement |
| CUDA out-of-place → COW | `tiny`, `long_short` | 1.241×, 1.374× | Every process pair improves |
| CUDA out-of-place → COW | `large_16` | 0.963× | Every pair regresses, 0.956–0.975× |
| CPU scalar → vector | `large_1`, `large_4`, `large_16` | 15.572×, 5.893×, 5.576× | Every pair improves |
| CPU scalar → vector | `tail_minus`, `tail_plus` | 13.287×, 14.298× | Every pair improves; includes the scalar tail launch |
| CPU scalar → vector | `tiny` | 1.655× | One pair is 0.979×; no uniform small-workload gain |

For sixteen large parameters, the reuse probe observes 48/48 outputs reused with exclusive
ownership and 0/48 with shared inputs; the disabled-COW executable reuses 0/48 in both cases.
The direct CPU pool peak sum falls from 448 MiB without COW to 328 MiB with COW, while live
allocations and live bytes after the step remain 64 and 256 MiB, including retained gradients.
This demonstrates reduced temporary allocation pressure, not a reduction of persistent state.
CUDA `large_16` still has a throughput regression despite correct reuse.

Module Foreach is measured against the existing Fused path, using three processes of 40
alternating pairs. The full optimizer-step-plus-sync results include collection and writeback:

| CPU workload | Fused median / p95 ms | Foreach median / p95 ms | Ratio | Per-process ratios |
| --- | ---: | ---: | ---: | --- |
| `tiny` | 0.9626 / 1.4026 | 1.0024 / 1.3999 | 0.960× | 1.105, 0.955, 0.976 |
| `mixed_rank` | 0.1002 / 0.3652 | 0.1423 / 0.3595 | 0.704× | 0.572, 0.628, 1.311 |
| `large_1` | 0.1406 / 0.3786 | 0.1397 / 0.3820 | 1.007× | 0.989, 0.981, 1.024 |
| `large_4` | 1.6442 / 2.6795 | 1.6373 / 3.0702 | 1.004× | 0.988, 1.014, 0.998 |
| `large_16` | 7.7084 / 11.9648 | 7.7953 / 11.7230 | 0.989× | 0.976, 0.983, 1.031 |
| `near_limit` | 0.3738 / 0.5636 | 0.3940 / 0.4330 | 0.949× | 0.945, 0.952, 0.956 |
| `long_short` | 1.9033 / 4.7896 | 1.8308 / 4.6986 | 1.040× | 1.038, 1.009, 1.034 |
| `over_limit` | 16.6381 / 24.5013 | 2.3249 / 4.4235 | 7.156× | 6.982, 7.179, 7.002 |
| `fragmented_lr` | 16.1024 / 24.6150 | 16.1675 / 20.8825 | 0.996× | 1.039, 0.992, 0.987 |

Only the eighty-compatible-small-parameter case shows a large, repeated Foreach gain here:
80 single launches become 10 batches. `tiny` becomes four batches but is slightly slower in
aggregate; the three-tensor heterogeneous case regresses. Prefer Fused for small heterogeneous
modules and measure Foreach on the actual parameter distribution. Near-limit items retain an
approximately 5% collection-attempt cost even though they are refused before deferred execution.
CUDA controls use no batch launches: pooled ratios are 0.996–1.004×, with per-process ratios
0.991–1.022×, consistent with their shared single-parameter path.

First-use and code-generation costs matter separately. In one fresh process per mode, CPU
`tiny` first-step latency is 15.204 ms Fused versus 98.939 ms Foreach; `over_limit` is 35.155
versus 62.720 ms. These are observations with the available JIT cache, not a cold-cache bound.
Both module modes retain the same reported live bytes and pool peak sum for the measured
workloads; that snapshot does not count transient allocation churn. Instrumented CPU
`over_limit` windows report a `system` timing method and medians of 16.211 versus 2.285 ms.
CUDA uses `device` timing. Neither profile series is substituted for uninstrumented timings.

The inspected x86-64 kernel objects contain 409/445 bytes of `.text` for scalar initial/stateful
single updates, 537/602 bytes for vectorized single updates, and 10,028/13,385 bytes for
eight-tensor Foreach updates. Vectorization increases this two-variant code total from 854 to
1,139 bytes (33.4%); batching's two observed variants total 23,413 bytes. These are executable
section sizes, not LLVM text-file lengths, total JIT memory, or a bound on all shape variants.
The eighty equal-shaped parameters reuse those two Foreach variants; different compile-time
chunk descriptors can produce more. No process-wide compilation-memory claim is made.

## Confirmation on the reviewed source

The rebuilt candidate is `9027a164e8a6db51f27b3baecc26123bef1a4e1a3a78a58178030b9a54357dd9`;
its scalar ablation is `b376d794bebbbf5abcc01c0a4624ef6967566e38a154f6ebd1be818a92aa89dc`.
All 184 source hashes covered by the build manifest match commit
`42df33e245d8a43e0010f2c5fe978cf184c8afd0`. Its library sources are identical to the published
core commit used by the downstream consumer; the later commit adds the offset benchmark.
The scalar ablation differs only in the vector-width selection condition, with the exact patch
preserved in the evidence. These confirmation runs are separate from the broader series above.

| Comparison | Workload | Baseline median / p95 ms | Candidate median / p95 ms | Median ratio | Per-process ratio range |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU Reference → Fused | `large_16` | 112.7220 / 152.3729 | 7.0687 / 9.5671 | 15.947× | 15.781–16.542× |
| CUDA Reference → Fused | `large_16` | 1.3370 / 1.3527 | 0.8844 / 0.9606 | 1.512× | 1.504–1.528× |
| CPU Fused → Foreach | `over_limit` | 16.5224 / 24.7717 | 2.2583 / 4.4999 | 7.316× | 6.883–7.564× |
| CPU Fused → Foreach | `fragmented_lr` | 16.6285 / 24.6943 | 16.6298 / 31.9908 | 1.000× | 0.994–1.012× |
| CUDA Fused → Foreach control | `over_limit` | 0.7488 / 0.8652 | 0.7460 / 0.8314 | 1.004× | 0.975–1.025× |
| CUDA Fused → Foreach control | `fragmented_lr` | 0.7496 / 0.8335 | 0.7492 / 0.9897 | 1.000× | 0.978–1.005× |
| CPU scalar binary → candidate | `unaligned` | 1.8438 / 2.6996 | 1.8761 / 3.6141 | 0.983× | 0.911–1.080× |

The module runs use three processes of 40 alternating pairs; direct runs use three processes
per mode with 30 samples each; the scalar-offset comparison uses four alternating process
pairs. Equal medians do not imply equal tail latency: the fragmented CPU groups have a higher
observed Foreach p95. CUDA controls still submit single-parameter updates.

The `unaligned` workload gives the gradient a four-byte offset on CPU/CUDA. Both the scalar
ablation and candidate use that same view; the candidate must select its scalar fallback, so
there is no vectorization benefit to claim. Numerical/path verification uses an independently
allocated, aligned Reference tensor with identical logical values because the generic vector
path assumes aligned bindings. The direct Reference/Fused measurements therefore compare
different physical layouts:

| Alignment control | Aligned Reference median / p95 ms | Offset Fused median / p95 ms |
| --- | ---: | ---: |
| CPU | 1.0301 / 1.2869 | 1.9111 / 3.9303 |
| CUDA | 0.1053 / 0.1541 | 0.0268 / 0.0499 |

The CPU offset update is slower than the aligned reference control. Neither row isolates a
same-layout fusion speedup; the scalar-ablation comparison above is the matching-layout
control for the vector policy.

## Evidence and remaining limits

The earlier series in the durable evidence contains 20 reports, 419 measurement processes,
10,540 sample rows and six binary manifests. The 2026-11-08 snapshot is recorded separately in
`upstream_sync`. Validation failures and successful reruns, source identities and agent review
records accompany the measurements.

| Requirement | Evidence and limit |
| --- | --- |
| Single-parameter fusion | Actual specialized launches and multi-step parameter/all-state comparisons cover decay variants, layouts and special values on the recorded CPU/CUDA configurations. Specialized execution remains limited to eligible raw F32 inputs. |
| COW reuse | Independent reuse/alias tests, out-of-place ablations, allocation/peak-memory measurements and shared-input controls passed. Shared storage cannot be reused; CUDA `large_16` throughput regresses. |
| Throughput | Scalar/vector comparisons cover 1/4/16 large parameters, tails and small controls; matching-offset controls show no unaligned vector benefit. JIT/code-size and first-use costs are recorded separately. |
| Foreach | Real module batches, scalar/group/stream boundaries, state/context preservation and end-to-end controls are exercised. Only compatible small CPU parameters batch; heterogeneous modules can regress. |
| Fallbacks | Flex, raw/fusion, tracked/unsupported input and actual shard/remote/Capture tests passed as recorded. Capture evidence concerns IR equivalence; CUDA hardware graph replay and unavailable runtime hardware are not established. |
| Integration | The 2026-11-08 snapshot passed focused checks and complete `cargo run-checks`; standalone downstream consumption remains verified only on the earlier pins. Upstream integration of the fork dependency prerequisites is still required before merge. |
| Review and delivery | The recorded source commits each have at least two agent reviews before commit, with findings rechecked. [Draft PR #6437](https://github.com/tracel-ai/burn/pull/6437) includes explicit AI disclosure; no human review is claimed. |

Current Linux validation uses an Intel Core i9-14900HX and NVIDIA GeForce RTX 5070 Ti Laptop
GPU, driver 615.71.10, with Rust 1.99.1. Hardware, driver, OS, Rust/Cargo versions, backend,
features, build profile, source hashes, and binary hash accompany the measurements above.

## Building and measuring the exact executable

Run the build helper from the repository root on a stable source tree:

```sh
python3 docs/optim-adamw/build.py target/adamw-evidence/adamw-final \
  --features std,cpu,cuda
```

`build.py` disables both Rust wrappers, builds the bench profile with `--locked`, and saves
the executable, compiler log, and `<binary>.json`. Its manifest binds the binary SHA256 to
the build command, feature set, Rust/Cargo versions, flags, Git HEAD, and a specific source
subset: root `Cargo.toml`, `Cargo.lock`, `.cargo/config.toml`, and all Rust files plus manifests
under `burn-optim` and `burn-cubecl`. It hashes that subset before and after compilation and
rejects a changed subset. This is not a whole-workspace fingerprint: other local crates,
measurement scripts, generated files, environment configuration, and dependency source trees
are not hashed. Git HEAD does not describe uncommitted changes outside that subset. Keep the
whole source tree stable and retain the environment/dependency record with accepted results.

Use the saved binary and its manifest together. `measure.py` verifies the executable hash
and feature set before recording commands, raw output, environment and load observations.
For the incremental module comparison against the existing single-parameter Fused path:

```sh
python3 docs/optim-adamw/measure.py target/adamw-evidence/adamw-final \
  target/adamw-evidence/module-cpu.json --backend cpu --features std,cpu,cuda \
  --module --repeats 3 \
  --workloads tiny mixed_rank large_1 large_4 large_16 near_limit long_short over_limit fragmented_lr
python3 docs/optim-adamw/summarize.py target/adamw-evidence/module-cpu.json
```

The module workloads deliberately include both admitted batches and fallback controls:

| Workload | Parameters | Expected raw CPU Foreach path per step |
| --- | --- | --- |
| `tiny` | 30 matrices of 64 × 64 | 4 batches |
| `mixed_rank` | A 64-vector, 64 × 64 matrix, and 16 × 8 × 3 × 3 tensor | 1 batch |
| `large_1`, `large_4`, `large_16` | 1, 4, or 16 matrices of 1,024 × 1,024 | 1, 4, or 16 single updates |
| `near_limit` | 8 matrices of 512 × 256, each filling the chunk budget | 8 single updates, refused before collection |
| `long_short` | 30 small matrices plus 2 large matrices | 4 batches and 2 single updates |
| `over_limit` | 80 matrices of 64 × 64 | 10 bounded batches |
| `fragmented_lr` | 80 small matrices, each with a different learning rate | 80 deferred single updates |

Module measurement defaults to `paired`: each fresh process creates independent Fused and
Foreach models, warms each for eight steps, asserts the actual batch/single launch path, then
records 40 pairs with alternating execution order. Three repeats retain three process runs;
they are not interchangeable with 120 independent machine-level replicates. Gradient
construction and its synchronization are outside the timed interval. The enqueue interval
includes the whole module optimizer step, including learning-rate cloning, traversal,
grouping, allocation, submission and writeback; total time adds the final synchronization.
Analyze enqueue and total medians/p95 separately and retain pair/repeat identity.

Measure first use, allocator usage and teardown in separate processes, and instrument device
windows separately from the uninstrumented paired timing:

```sh
python3 docs/optim-adamw/measure.py target/adamw-evidence/adamw-final \
  target/adamw-evidence/module-cpu-lifecycle.json --backend cpu --features std,cpu,cuda \
  --module --repeats 1 --modes fused foreach \
  --workloads tiny large_1 large_4 large_16 near_limit over_limit fragmented_lr
python3 docs/optim-adamw/measure.py target/adamw-evidence/adamw-final \
  target/adamw-evidence/module-cpu-profile.json --backend cpu --features std,cpu,cuda \
  --module --repeats 1 --modes profile-fused profile-foreach \
  --workloads tiny large_1 near_limit over_limit fragmented_lr
```

The direct equivalents are `BURN_DEVICE=cpu <binary> --module-samples paired tiny` and
`BURN_DEVICE=cpu <binary> --module-samples profile-foreach tiny`. Lifecycle mode emits first-step, allocator, verified
path, 30 steady samples, and final model/optimizer drop records. Its pool-peak sum is the sum
of per-pool historical maxima, not necessarily simultaneous process peak memory. First-step
latency includes JIT work available to that process; it is not guaranteed to start with an
empty persistent compilation cache. Profile mode warms for eight steps and records 20 windows
with the runtime's reported timing method plus host elapsed time. Those markers can perturb
timing, so profile windows are diagnostic and must not be mixed into paired speedup samples.

For the direct Reference/Fused benchmark, omit `--module`; `--shared` retains external input
handles to measure COW behavior. Foreach is meaningful as a module path. CUDA module runs are
fallback controls, since CPU is the only runtime admitted for batching in this implementation.

Measure compilation separately from steady state. Compare initialized independent inputs,
warm both modes, alternate execution order, synchronize at equivalent boundaries, and retain
raw samples. Include enqueue and full step-plus-sync median/p95, allocation/live/peak memory,
first-step latency, and actual launch counts. For incremental claims, change only the feature
being evaluated; cumulative Reference-to-candidate gains cannot establish each stage's benefit.
Run benchmarks serially on an otherwise idle machine, without concurrent builds. A faster
kernel is insufficient if module traversal, metadata, allocation, or synchronization makes the
whole step slower. Report regressions and restrict runtime eligibility when the evidence does
not support enabling a path.

The adjacent measurement scripts retain their separate purposes: `build.py` captures a binary
and source subset, `measure.py` records that binary's runs, `compare.py` alternates two binaries
with matching source manifests, and `summarize.py` summarizes recorded observations. A source
manifest must be captured with the exact built binary; hashing the current checkout after a
different build is not provenance.
Historical drafts, patch archives, and superseded measurements are recoverable from git
history; they are not part of the current result set.
