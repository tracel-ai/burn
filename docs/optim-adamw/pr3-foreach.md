# PR 3：AdamW 模块级多参数 Foreach

> 文档状态：独立 PR 设计稿，尚未实现或测量
>
> 性能类型：减少多参数模型的物理 launch 与 host enqueue 开销
>
> 代码依赖：单参数 Fused 已合并；COW 是可选复用，不是前置条件
>
> 外部前置：CubeCL 能查询 pinned/unpinned client 的有效执行 stream；当前 revision 不具备时先提供只读 API
>
> 直接性能对照：当前 `main` 上最好的逐参数 Fused；Reference 只作辅助对照

本文独立说明该 PR 的实现、正确性和性能门禁，不需要阅读其他 PR 文档。

## 1. 性能问题与独立价值

即使每个参数已经使用单参数 Fused，微小多参数模型仍需要逐参数 host 元数据处理和内核提交。Foreach 把多个不同 rank、不同 shape 的独立张量交给一次或少数几次多参数内核，以减少 launch 和 enqueue。

Foreach 不复制或拼接整个模型，也不进一步降低每元素逻辑 memory traffic。对无 AMSGrad、已初始化参数，Fused/Foreach 都需要：

```text
read:  param, grad, old_m1, old_m2 = 4 个 tensor streams
write: new_param, new_m1, new_m2   = 3 个 tensor streams
```

AMSGrad 有 old max 时增加 read/write；首步不读旧 state。Foreach 的独立收益只能来自 launch/host 开销下降，且必须覆盖两次 module map、metadata、输出分配和 JIT 成本。

本 PR 一次交付完整模块接入、分组、多参数 kernel、结果与 state 写回、上限拆分和性能证据，不设置“prototype PR”或“模块接入 PR”两个无独立收益阶段。

## 2. 范围与兼容边界

本 PR 包含：

- 收集合格 AdamW 参数，保存完整执行上下文；
- 按 device/service/stream/dtype/lr/time/variant 分组；
- 一维 chunk 多参数内核和 launch 上限；
- 单 item、空组、失败回退和有效 stream 恢复；
- checkpoint、decorator、alias、chunk 和模块恢复测试；
- 与逐参数 Fused 的直接 A/B。

本 PR 不包含：

- 全模型物理拼接或矩形 `MaxSmallLen × N` 网格；
- 改变参数组、逐参数 lr/time、缺梯度或 clipping 语义；
- 全局 gradient clipping、FP32 master weights 或 checkpoint 格式变化；
- Fusion CustomOpIr、dtype 扩展或大参数向量化；
- 若 COW 尚未合并，额外实现另一套 COW planner。

现有 AdamW 默认值保持 `beta_1=0.9`、`beta_2=0.999`、`epsilon=1e-5`、`weight_decay=1e-4`，AMSGrad/cautious 默认 false。Reference 始终保留，只有显式选择 Foreach 才尝试批处理。

### 2.1 执行配置与公开 API

单参数 Fused 引入的执行枚举已经是 `#[non_exhaustive]`。本 PR 与真实模块级实现一起增加 Foreach，不提前公开空壳 variant：

```rust
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum AdamWExecutionConfig {
    Reference,
    Fused,
    /// 优先执行模块级多参数融合；单项或不支持时使用 Fused/Reference。
    Foreach,
}
```

`AdamWConfig.execution` 仍为 Option：缺省/旧配置是 Reference，显式 Foreach 序列化和反序列化后保持 Foreach。直接调用单参数 `Optimizer::step` 时没有模块批次上下文，Foreach 使用单参数 Fused；模块路径中只有实际满足本 PR 全部准入条件才提交多参数 launch。

本 PR 增加 Foreach 配置 round-trip、旧配置仍默认 Reference、直接 step 的单项行为和模块级路径测试。`#[non_exhaustive]` 避免下游 exhaustive match 因新增 variant 被破坏；如果合入前代码不具备该属性，必须先解决 public enum 兼容性，不能静默引入 breaking change。

## 3. 数值与状态契约

多参数内核逐元素执行与单参数 Reference 相同的公式，不能因跨参数处理改变顺序。

Host 对每个参数分别计算：

```text
time_i32   = state.time as i32
factor_1   = 1.0f32 - beta_1
factor_2   = 1.0f32 - beta_2
bias_1     = 1.0f32 - beta_1.powi(time_i32)
bias_2     = 1.0f32 - beta_2.powi(time_i32)
decay_rate = lr * (weight_decay as f64)
```

已有 state 时，按 tensor dtype F：

```text
m1 = old_m1 * F(beta_1) + g * F(factor_1)
m2 = old_m2 * F(beta_2) + (g * g) * F(factor_2)
```

首步不读取旧 state：

```text
m1 = g * F(factor_1)
m2 = (g * g) * F(factor_2)
max_m2 = amsgrad ? m2 : absent
```

已有 state 时：

```text
if amsgrad:
    previous_max = has_old_max ? old_max : m2
    new_max = (previous_max < m2) ? m2 : previous_max
    v = new_max
else:
    new_max = has_old_max ? old_max : absent
    v = m2

m1_corrected = m1 / F(bias_1)
v_corrected = v / F(bias_2)
delta = m1_corrected / (sqrt(v_corrected) + F(epsilon))
```

decay：

```text
NoDecay:  decayed = p
Standard: decayed = p * F(1.0f64 - decay_rate)
Cautious: decayed = p - (((p >= 0) == (m1 >= 0)) ? p * F(decay_rate) : F(0))
p_new = decayed - delta * F(lr)
```

关键约束：

- 每个参数使用自己的 lr、time、bias、超参数和 state；不能因同组而共享不同数值。
- bias 是分母，不改成 reciprocal 乘法，也不使用 Keras alpha 公式。
- NoDecay 不引入 `0*Inf`；Cautious 不改成 `(p*m1)>=0`，并使用更新后的 m1。
- AMSGrad 保持 `old_max < m2` 的选择方向：old NaN 保留、new NaN 不覆盖有限 old、相等/signed zero 保留 old 位模式。
- 非 AMSGrad state→AMSGrad 时以新 m2 初始化 max；含 max state→非 AMSGrad 时保留 old max，但不更新、不用于 delta。
- clipping 仍在收集每个参数时按原逐参数规则执行，不能把 clipping 系数折进 lr。
- param/m1/m2/max 的 shape、rank、dtype、device、state time、RecordState/Burnpack 名称和 `__rank` 不变。
- 首版只支持 F32 非量化输入；F16/BF16/F64/Flex32、量化、不同 state dtype、空张量、tiled 和不支持 layout 回退 Reference。
- 非连续输入由每个 LinearView 自己的 metadata 寻址，不强制拼接、reshape 或 D2H sync。
- F32 是容差兼容而非 bitwise 承诺；特殊值另做分类或位模式断言。

半精度不能顺带开放：CubeCL Chain/Accumulators 可能把一串 F16 算术保持在高精度，而 Reference 每个 tensor op 有 dtype 物化边界。例如 F16 首步 `grad=300,beta_2=0.999`，Reference 的 square 先物化为 Inf，fused 高精度链却可能得到有限 m2，两者语义不同。

## 4. 两次 module map

当前 `ModuleMapper::map_float` 必须立即返回一个参数，而 raw Cube kernel launch 在调用时就已排队，不能先返回参数再收集未来 batch。因此本 PR 使用：

```text
module.map(CollectMapper)
    ├─ 无梯度：原样返回，不推进 time
    ├─ 非 AdamW/非 Foreach/不合格：立即执行原 step_dyn
    ├─ rank 不在 dispatch_rank! 支持表：立即执行原 step_dyn
    └─ 合格 AdamW：保留原参数，移除梯度，生成 PendingAdamW
                  ↓
        partition + chunk + launch
                  ↓
module.map(ApplyBatchMapper)
    └─ 按 ParamId 替换参数，恢复 ParameterContext 和 ParamMapper
```

两次遍历只做 host 元数据操作，不做 tensor 打包或 D2H 同步，其成本必须计入 benchmark。

原 mapper 的语义保持不变：

1. Param::consume 取得 ParamId、tensor 和 ParamMapper；
2. 无梯度不更新、不推进 time；
3. 按“最后匹配参数组”选择 optimizer/clipping；
4. clipping 先于 optimizer 更新；
5. 逐参数求 lr；
6. 延迟并批量执行 inner backend 更新；
7. 恢复 require_grad、distributed、gradient-checkpointing 策略；
8. 恢复 mapper/reparameterization 信息。

### 4.1 Rank 准入

rank 准入先于 primitive 转换、梯度移除和 StateBuilder 创建。当前 `dispatch_rank!` 只支持 0–8；D>8 在收集阶段调用原 step_dyn，由同一支持表按当前行为立即 panic。

不得改用泛型 reference_step 让 rank 9 首次更新成功，再把失败推迟到 `to_device_dyn`、state flatten/unflatten。若未来支持更高 rank，必须同时扩展所有动态路径，而不是只放宽 Foreach。

### 4.2 发现具体 AdamW

`ModuleOptimizer` 存储 `Arc<dyn DynOptimizer>`。可以增加默认返回 None 的 hidden capability：

```rust
#[doc(hidden)]
fn as_any(&self) -> Option<&dyn core::any::Any> { None }
```

现有 blanket `impl<O: Optimizer> DynOptimizer for O` 返回 Some(self)；自定义 DynOptimizer 默认没有 foreach 能力。收集器只在 `downcast_ref::<AdamW>()` 成功且 execution 为 Foreach 时建立 pending。

这不是 backend capability 查询；backend 是否可执行仍由 primitive 边界决定。

## 5. Pending 与状态恢复

```text
PendingAdamW
├─ ParamId 与用于写回 state 的原 map key
├─ optimizer 完整数值配置和 clipping 后梯度
├─ lr: f64
├─ rank（仅用于恢复）
├─ param/grad/m1/m2/max_m2 primitives
├─ has_state、has_old_max、原/新 time 和 bias 分母
├─ device、client service、有效执行 stream、dtype、numel、layout/address 信息
├─ optimizer Arc、path、grad_clipping（恢复 OptimizationContext）
└─ StateBuilder
```

StateBuilder 在泛型 `map_float<const D: usize>` 中、动态 rank 准入后创建，可表示为 `FnOnce(RawAdamWState) -> DynState`：把 primitive 重新构造成 `AdamWState<D>` 并调用 `DynState::create(..., D)`。这样不需要不存在的 `Vec<AdamWState<D>>`，也不丢失 `__rank`。

ApplyBatchMapper 再次 Param::consume，按 ParamId 取输出，调用 ParameterContext capture/restore 和 Param::from_mapped_value。没有输出的参数原样返回。path、mapper 和 state 必须各恢复一次。

转换失败必须保留原输入。对 Tensor 做句柄级 clone 后尝试 primitive 转换；任一失败则 drop 已转换 clone，用完整原输入在该项保存的 stream 中走 Reference。所有回退都发生在 launch 前，不能部分执行后换算法重试。

## 6. 有效执行 stream 是启用门禁

Pending 的 stream 是收集时原 step_dyn 本应使用的**有效执行 stream**，不是 tensor allocation 的创建 stream。unpinned client 来自当时的 `StreamId::current()`，显式 pinned client 来自固定 stream；`Handle.stream` 不能用于这个判断。

当前锁定 CubeCL 的 `ComputeClient::stream_id()` 已实现 `Some(pinned)->pinned / None->StreamId::current()`，但方法私有。本 PR 实施前必须在指定 CubeCL revision 提供等价只读 API，例如：

```rust
pub fn execution_stream(&self) -> StreamId {
    self.stream_id()
}
```

burn-cubecl 边界在 CollectMapper 仍位于原执行作用域时查询并保存该值，同时保存 `client.service_id()`。若 revision 无法查询 pinned/unpinned 有效 stream，本 PR 不得启用 Foreach；不能假定 client 未 pinned，也不能改用 Handle.stream。

这个上游只读 API 是正确性前置条件，不单独包装成一个没有 Burn 性能收益的 PR 来充数。

每组全部输出分配、单 item Fused、所有 chunk launch，以及收集后才确定的 Reference 回退，都必须置于保存的有效 stream 中：

```rust
for group in groups {
    group.execution_stream.executes(|| {
        // 用同 service_id 的代表 client 分配本组输出；
        // 在该作用域提交单项或 chunk launch；
        // delayed Reference 也使用各自保存的 stream。
    });
}
```

即使外层默认 stream 从 A 切到 B，A 上收集的参数仍在 A 更新，A 上后续消费者保持排队顺序，输出 allocation 也归属正确 stream。观测 snapshot 同时记录实际 launch 的 service/stream，测试不能只检查分组 key。

## 7. 分组键与失败边界

首版按完整 key 分组：

```text
(concrete Cube device, client service_id, effective execution stream id, dtype,
 lr.to_bits(), state.time / time_i32,
 beta_1.to_bits(), beta_2.to_bits(), epsilon.to_bits(), weight_decay.to_bits(),
 has_state, has_old_max, amsgrad, decay_mode)
```

原因：

- 不同 device、client service 或有效 stream 不能为减少 launch 改变顺序；
- ModuleLearningRate 可以逐参数不同，缺梯度或加载历史会让 time 不同；
- NaN 超参数用 bits 分组，避免 `NaN != NaN` 导致异常分派；
- state/max/AMSGrad/decay 是不同 kernel variant，不能共用错误读取路径。

rank 和 shape 在当前动态支持的 0–8 内不必相同；非连续 layout 由每个 view 自己的 metadata 描述。首版只支持同 dtype 组。未来 per-tensor scalar list 是另一项需要独立收益证明的优化，不在本 PR 顺带加入。

失败边界：

- 空组直接返回；只有一项时复用单参数 Fused，但仍在保存的 stream 中分配输出并 launch；
- 收集时已发现不合格立即走原 step_dyn；收集后才确定的回退在该项保存的 stream 中执行；
- 分组/chunking 可在外层准备，输出分配必须在对应 stream 作用域内、launch 前完成；
- 若没有 COW，本 PR 全部 out-of-place，因此不同参数或 state 共享 allocation 也不会写坏输入；
- 若已有 COW，只复用既有 planner，不再写一套所有权规则。

## 8. 多参数 chunk kernel

### 8.1 一维 chunk 列表

不做矩形 `MaxSmallLen × N` 网格，以免短 tensor 的线程空转：

```text
host:
    cube_dim = 一维，N = 1
    chunk_len = cube_dim.x
    对每个 tensor：
        chunks = ceil(numel / chunk_len)
        为每个 chunk 记录 tensor_index、local_offset、valid_len
    cube_count.x = 本批全部 chunk 数
```

每 chunk 对应一个 cube，线程局部位置 `local=UNIT_POS_X`，逻辑索引为 `index=local_offset+local`。只处理 `local<valid_len`；所有 param/grad/state read 和 output write 使用同一个 index，LinearView 不会自动注入 chunk offset。

大 tensor 自然占多个 chunk，小 tensor 只占一个，无需固定 65,536 阈值或按大小分两桶。

### 8.2 Sequence 的编译期展开

CubeCL `Sequence<T: LaunchArg>` 可以注册动态数量参数，但索引和迭代在编译期展开，不能像 PyTorch 一样用运行时 tensor_id 直接索引。因此使用编译期 chunk descriptor 和 uniform 分支：

```rust
#[unroll]
for chunk in 0..args.chunks.len() {
    if CUBE_POS_X == chunk {
        let t = comptime![args.chunks.index(chunk).tensor_index];
        let offset = comptime![args.chunks.index(chunk).local_offset];
        let local = UNIT_POS_X;
        if local < comptime![args.chunks.index(chunk).valid_len] {
            let index = offset + local;
            // 所有 read/write 统一使用 index。
        }
    }
}
```

每个 cube 的 CUBE_POS_X 在 cube 内一致，不会产生 cube 内按元素搜索 tensor 的发散。

本 PR 开发初期先用 2～3 个 tensor 做 `Sequence<LinearView>` compile-only proof。proof 只是实现步骤，不是独立 PR；若 API 组合不成立，停止或修改设计，不提交不可编译原型，也不退回物理拼接。

### 8.3 Launch 上限与 JIT

每次 launch 的参数和展开代码随 tensor/chunk 数增长，必须设置：

- `max_tensors_per_launch`；
- `max_chunks_per_launch`；
- 必要时 `max_unrolled_elements_metadata`，限制 Host descriptor 大小。

常量不能照抄 PyTorch 的 110/320 或 65,536。先用较小值完成正确性，再按 CUDA/ROCm/Metal/WGPU/CPU 参数限制和编译耗时选择。超过上限就拆分，不承诺全模型固定 launch 数。

JIT key 包含 dtype、variant 和 chunk descriptor。shape 稳定时后续 step 应复用编译结果，benchmark 分别报告首次编译和稳态，防止每步重新 JIT 被误算成融合收益。

## 9. 执行模式与 checkpoint

| 模式 | 本 PR 行为 |
|---|---|
| Reference | 原实现，不收集 batch |
| raw Cube eager | 合格参数进入 Foreach |
| 单参数 `Optimizer::step` | 没有模块批次上下文，使用单参数 Fused |
| Autodiff | 保留 inner tensor 和 ParameterContext 恢复，不建反向节点 |
| Fusion | 不伪造 CustomOpIr，回退 Reference |
| Remote / Capture / Group | primitive 不匹配时回退 Reference |
| 量化 / 不支持 dtype/layout | 回退 Reference |
| `step_multi` | 先沿用 GradAdaptor 梯度合并/device 结果，再按有效 stream 分组提交 |

checkpoint 双向验证：Reference 跑 N 步保存→Foreach 加载继续，以及 Foreach 保存→Reference 加载继续。覆盖 AMSGrad 开关和无 max↔有 max 迁移，检查每个 ParamId 的 `__rank`、time、tensor 名称、shape、dtype、group/path 和逐参数 lr。浮点 state 按容差比较，不要求 bitwise 相同。

## 10. 正确性测试

raw 测试先断言 `!burn_cubecl::FUSED`，路径计数只在实际提交专用 launch/chunk 后增加。观测计数使用测试锁和 Device::sync 隔离 before/after；Fusion 测试断言专用计数不增加。

除单参数数值矩阵外，本 PR 至少覆盖：

| 类别 | 用例与断言 |
|---|---|
| 混合组 | 同组不同 shape/rank，混合 rank 模型正确恢复 |
| 参数组 | 不同 AdamW 配置/lr，验证“最后匹配组”不变 |
| 缺梯度 | 参数和 time 不变，其他参数正常更新 |
| 分组 | 同 shape 不同 lr/time 不合并；同超参数但 state 初始化状态不同 |
| rank | 0 和 8 正常；9 在原 step_dyn 边界失败，launch=0、无 pending/state |
| stream | 同 service 不同有效 stream 不合并；A 收集、外层切 B 后仍在 A 分配和 launch |
| pinned | 显式 pinned client 正确；无法查询有效 stream 时 Foreach→Reference |
| 空/单项 | 空 batch 无 launch，单 item 走单参数 Fused 且 stream 正确 |
| chunk | 长短 tensor、跨多 chunk、尾 chunk；`numel=2*chunk_len+3` 逐元素恰好更新一次 |
| 上限 | 超过 tensor/chunk 上限正确拆分，所有元素恰好更新一次 |
| JIT | chunk descriptor 改变时 key 不碰撞，稳定 shape 可复用编译 |
| alias | out-of-place 下共享 allocation 不覆盖输入；若有 COW，复用同一 alias 矩阵 |
| module | step_multi、Autodiff context、reparameterized 参数各恢复一次 |
| decorator | Fusion、Remote、Capture、Group 回退，不触发不支持 panic |
| checkpoint | Reference↔Foreach 双向恢复及 AMSGrad state 迁移 |
| 数值 | 首步/多步、Cautious、AMSGrad NaN/signed zero、decay 和 clipping |

Reference 和 candidate 在同一 device、同一 dtype 下比较，避免跨后端差异。

## 11. Benchmark 与性能验收

### 11.1 测量对象

至少包含：

- 30 个 `[64,64]` F32 微小模型；
- bias `[64]`、linear `[64,64]`、conv `[16,8,3,3]` 混合 rank；
- 多个小 bias 与 1～4 个 `[1024,1024]` 大矩阵的长短混合；
- 1、4、16 个 `[1024,1024]` 大参数；
- 超过 tensor/chunk 上限的模型；
- 不同 lr/time、不同 stream 和单 item 对照。

分别报告：

1. Host enqueue，计时区不 sync；
2. enqueue + Device::sync 总时间；
3. 两次 module map 和 metadata 构建成本；
4. 输出分配与 step 结束后的 drop/sync；
5. 首次 JIT 与稳态；
6. launch 数、批次数、chunk 数、tensor 数和原始结果；
7. median/p95、warmup、样本数、hardware、runtime 和 features。

直接 A/B 两边都使用当前逐参数 Fused，只有本 PR 的 Foreach 不同。Reference 数据只作辅助背景。若已有 COW，A/B 两边保持 COW 配置相同，必要时另报 COW on/off，不能把 COW 收益归给 Foreach。

### 11.2 合并门禁

- launch-bound 小参数模型在计入两次 module map 后总时间有净收益；
- enqueue、device、drop、metadata、JIT 和 launch/chunk 数均有拆分数据；
- 大参数吞吐不得因 unroll/分支/chunk 退化；
- 超过上限的模型没有失控 JIT 或参数大小；
- 长短混合 workload 没有不可接受回退；
- 收益只覆盖实测 runtime，不外推所有平台。

若两次遍历、metadata 或 JIT 持续抵消收益，则不合并本 PR；不影响已经独立交付的其他优化。

## 12. 验证命令

按实际可用 runtime 执行；不可用硬件明确标为 skipped。

```sh
# Reference、feature isolation 与仓库回归
cargo test -p burn-optim --features burn-core/flex
cargo test -p burn-optim --no-default-features --features std,fusion,burn-core/flex
cargo run-checks

# CubeCL CPU raw、Fusion 与路径/chunk 观测
cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,cpu,fusion,adamw-bench-observability

# benchmark 先验证路径，再无观测开销计时
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu,adamw-bench-observability -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu

# CUDA 示例；ROCm/Metal/WGPU 按实际硬件执行对应 raw/Fusion 测试
cargo test -p burn-optim --no-default-features --features std,cuda,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,cuda,fusion,adamw-bench-observability
cargo run-checks --backend cuda
```

`--verify-path` 必须验证预期 case 全部完成且非空，并打印实际 batch/chunk/service/stream；过滤成 0 case 或只启动 runner 都是失败。`cargo run-checks --backend ...` 不替代新增 optimizer 的 chunk、stream 和 checkpoint 测试。

检查后审阅生成 diff，确保 `Cargo.lock` 没有无关变化。未运行平台不得写成已验证。

## 13. PR 交付清单与停止条件

PR 描述至少包含：

- 多参数 launch/host 瓶颈和与逐参数 Fused 的直接 A/B；
- 两次 map、完整分组、有效 stream 和回退设计；
- chunk/上限/JIT 方案及 prototype 结论；
- 数值、rank、stream、checkpoint、decorator 和路径观测结果；
- enqueue/device/drop/metadata/JIT 数据；
- 实测平台、跳过平台、changelog 和 AI Usage。

出现以下情况停止本 PR：Sequence 参数形式不可行、无法查询并恢复有效 stream、必须引入 D2H sync、分组无法保留逐参数 lr/time，或端到端净收益不成立。某 runtime 不支持时保留 Reference，不用未验证代码兜底。

## 14. 实现参考

- `crates/burn-optim/src/optim/module/module_optimizer.rs`：参数组、clipping、lr、state 和恢复流程。
- `crates/burn-optim/src/optim/adamw.rs`：AdamW 数值与 state。
- `crates/burn-cubecl-fusion/src/engine/codegen/ir.rs`：Sequence 参数组织方式。
- 锁定 CubeCL `sequence/base.rs`：编译期索引和展开约束。
- [PyTorch AdamW](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)：foreach 分组和回退路径对照。
- [PyTorch CUDA MultiTensorApply](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh)：独立 tensor 地址、chunk 和 launch 上限；不照搬常量或运行时 metadata 索引。
