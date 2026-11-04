# AdamW 垂直融合与 Multi-Tensor Foreach 实施方案

> 状态：设计稿，尚未实现或测量
> 基线：Burn `ee39002d94e1c958d86a65e92132b3961d48c18c`
> CubeCL：`d88b0febb50e7b83a8548125f6a701c12f11a2ba`（`0.12.0-pre.2`）
> 关联文档：[CubeCL CPU 运行时优化](optim-adamw-cpu-runtime-optimization.md)

## 0. 结论与边界

本方案把两个容易混淆的优化拆成独立 PR：

1. **垂直融合（Fused）**：一次单参数内核完成一阶矩、二阶矩、可选 AMSGrad、偏差修正、AdamW 衰减和参数更新，消除中间张量。
2. **水平批处理（Foreach）**：把多个不同 rank、不同 shape 的独立张量交给一次或少数几次多参数内核，减少启动次数；不为它复制或拼接整个模型的参数。
3. **默认行为不变**：现有 `AdamWConfig` 的默认值、字段和参考算法保持原样。新增执行方式默认是 `Reference`，只有显式选择 `Fused` 或 `Foreach` 才尝试新路径。
4. **不扩大核心 `Backend` trait**：优先复用 `Tensor::try_into_primitive` 和 `burn-cubecl` 内核。Flex、Fusion、Remote、Capture、Group、量化输入及其他不支持的情况继续执行现有参考路径。
5. **不夹带其他算法改造**：不实现全局梯度裁剪、不引入 FP32 master weights、不改变现有逐参数 clipping、不改变 checkpoint 格式。
6. **CPU 调度器另案处理**：Burn 内的 Fused/Foreach 可以先减少 CPU 工作量；`cubecl-cpu` 的 inline、批量投递和内存回收属于上游独立改动，不能作为本方案的前置假设。

### 0.1 非目标

以下事项不在本方案中承诺：

- 全模型恒定 `1～3` 次物理 kernel launch；
- 所有设备、dtype、布局都使用专用内核；
- 与参考路径 bitwise 完全相同；
- “零分配”“100% SM 利用率”“无限接近 Flex”；
- 把 PyTorch 的默认值、 epsilon 位置或 Keras 的 `alpha` 公式直接复制到 Burn；
- 让 `Foreach` 改变参数组、逐参数学习率、缺梯度参数或 optimizer state 的语义。

## 1. 已核实的 Burn 基线

### 1.1 当前配置与调用路径

当前实现位于 `crates/burn-optim/src/optim/adamw.rs`：

| 项目 | 当前 main 的事实 | 本方案约束 |
|---|---|---|
| `beta_1` / `beta_2` | `0.9` / `0.999` | 不变 |
| `epsilon` | `1e-5` | 不变，不改成 PyTorch 的 `1e-8` |
| `weight_decay` | `1e-4` | 不变，不改成 `1e-2` |
| `amsgrad` | 已存在，默认 `false` | 融合实现必须支持 |
| `cautious_weight_decay` | 已存在，默认 `false` | 融合实现必须支持 |
| `grad_clipping` | 已存在 | 保留字段及逐参数语义 |
| `LearningRate` | `f64` | Host 计算精度不变 |
| state | 每个参数独立的 `AdaptiveMomentumState<D>` | rank、dtype、time、tensor 名称均不变 |

`ModuleOptimizerMapper::map_float` 当前按参数执行以下步骤：

1. `Param::consume()` 取得 `ParamId`、tensor 和 `ParamMapper`；
2. 从 `GradAdaptor` 移除梯度；没有梯度就不更新参数，也不推进 state time；
3. 按“最后匹配的参数组”选择 optimizer 和 clipping；
4. 在 `step_dyn` 之前执行 `GradientClipping::clip_gradient`；
5. 用 `ModuleLearningRate::lr_from_param` 求该参数的学习率；
6. 把 tensor 转为 `BridgeTensor`，在 inner backend 上更新；
7. 通过 `ParameterContext::restore` 恢复 `require_grad`、distributed 标记和 gradient-checkpointing 策略；
8. 用 `Param::from_mapped_value` 恢复 mapper/reparameterization 信息。

Fused 不得绕过第 6 步以外的任何语义；Foreach 也必须保留全部八步，只把第 6 步延后到分组完成后执行。

### 1.2 “15 次算子”只是逻辑数据流，不是普遍物理启动数

在无 AMSGrad、无 cautious decay、state 已初始化的参考路径中，可以把当前实现数成 15 个高阶 tensor 算子。这个计数有助于说明融合机会，但不能直接推导：

- 每种执行模式都发生 15 次物理 kernel launch；
- 每次都分配一个新缓冲区；
- 总显存访问恰好是 `15 read + 15 write`。

CubeCL 的 eager 算子会在允许时原地复用；Fusion 和 graph replay 还会改变物理执行方式。因此，“30 个参数就是 450 次物理发射”“Fused 必然减少 93% 启动”都只能作为待测假设。

### 1.3 逐参数 clipping，不是全模型 clipping

`crates/burn-optim/src/grad_clipping/base.rs` 当前实现是：

- `Value`：逐元素截断；
- `Norm`：对**每个参数分别**求 L2 norm；
- F16/BF16 先转 F32 计算 norm；
- 系数为 `threshold / (norm + min_positive)`，随后 `clamp_max(1.0)`；
- 最后乘梯度并 cast 回原 dtype。

本方案不在 AdamW 内核中重新实现或合并 clipping，避免同时改变作用域和公式。

## 2. 数值兼容契约

### 2.1 参考公式必须按 Burn 的顺序实现

令 `t` 为当前参数自己的 state time。Host 先按现有 Rust 代码计算：

```text
time_i32  = state.time as i32                 // 保持当前 cast 语义
factor_1  = 1.0f32 - beta_1                  // f32
factor_2  = 1.0f32 - beta_2                  // f32
bias_1    = 1.0f32 - beta_1.powi(time_i32)   // f32
bias_2    = 1.0f32 - beta_2.powi(time_i32)   // f32
decay_rate = lr * (weight_decay as f64)       // f64
```

已有 state 时，内核按 tensor dtype `F` 执行：

```text
m1 = old_m1 * F(beta_1) + g * F(factor_1)
m2 = old_m2 * F(beta_2) + (g * g) * F(factor_2)
```

首步不得读取未初始化的 old state，而是执行：

```text
m1 = g * F(factor_1)
m2 = (g * g) * F(factor_2)
max_m2 = amsgrad ? m2 : absent       // 仅限 state 原本为 None 的首步
```

已有 state 时，`max_moment_2` 是独立于 `has_state` 的状态：

```text
if amsgrad:
    previous_max = has_old_max ? old_max : m2
    new_max = (previous_max < m2) ? m2 : previous_max
    v_for_update = new_max
else:
    new_max = has_old_max ? old_max : absent   // 保留旧 Some，但不更新、不用于 delta
    v_for_update = m2

m1_corrected = m1 / F(bias_1)
v_corrected  = v_for_update / F(bias_2)
delta = m1_corrected / (sqrt(v_corrected) + F(epsilon))
```

因此必须同时记录 `has_state` 和 `has_old_max`。从非 AMSGrad state/checkpoint 切换到 AMSGrad 时，`has_state=true, has_old_max=false`，本次新 `m2` 同时充当 previous max 和结果，不能读取 dummy。反向切换时保留已有 `Some(old_max)` 及其原 buffer/值，不能擅自改成 `None`。

这里**禁止**用 `recip_bias_corr = 1 / bias` 后乘法替代除法，也禁止改成 Keras 的 `sqrt(bias_2) / bias_1` 形式。两者代数相近，但舍入顺序不同。

### 2.2 三种 decay 模式分别处理

Host 先根据 `decay_rate == 0.0` 选择模式：

```text
NoDecay:
    decayed = p

Standard:
    decayed = p * F(1.0f64 - decay_rate)

Cautious:
    p_nonnegative  = p >= 0
    m1_nonnegative = m1 >= 0
    apply_decay = (p_nonnegative == m1_nonnegative)
    raw_decay = p * F(decay_rate)
    decayed = p - (apply_decay ? raw_decay : F(0))
```

最后统一执行：

```text
p_new = decayed - delta * F(lr)
```

约束：

- `NoDecay` 必须跳过 decay 乘法，避免把参考路径中不存在的 `0 * Inf` 引入结果。
- Cautious 必须分别比较符号，不能改成 `(p * m1) >= 0`。例如 F32 的 `p=-1e-30, m1=1e-30`，乘积可能下溢为 `-0.0`，两种判断会不同。
- Cautious 使用**更新后的** `m1`，与当前 `momentum_state.moment_1` 一致。
- 不把 weight decay 乘进梯度或动量。

### 2.3 AMSGrad 复现 `max_pair`，不直接调用 `F::max`

当前 `old_max.max_pair(m2)` 等价于：

```text
mask = old_max < m2
max_m2 = mask ? m2 : old_max
```

专用内核必须使用同样的选择方向：

- old max 为 NaN 时，比较为 false，保留 old NaN；
- 新 `m2` 为 NaN 而 old max 有限时，比较为 false，保留 old max；
- 相等或 `±0` 比较为 false 时保留 old 的位模式；
- 首步或已有 state 但 `max_moment_2=None` 时直接以本次 `m2` 初始化，因此该情况下的 NaN 保留；
- `amsgrad=false` 时不执行上述 max 更新，也不把旧 max 用于 delta，但旧 `Option<max_moment_2>` 必须原样保留。

不能假定硬件 `max`、`fmax` 或 `F::max` 具有相同 NaN/signed-zero 规则。

### 2.4 精度、FMA 与容差

- Host 标量继续保持上述 f32/f64 精度；只在进入内核运算时转换为 `F`。
- 初版使用标量 `Const<1>` 运算，不做向量重排、不归约、不主动使用 `mul_add`、不做代数重组。
- 编译器仍可能进行 FMA contraction，且不同目标对除法、sqrt、subnormal 和 NaN 的指令选择不同，所以 F32 契约是**数值容差兼容，不是 bitwise 承诺**。
- 测试容差由输入 `dtype.finfo()` 推导。初始建议 `atol=4ε`、`rtol=8ε`，再按实际误差来源审查；不得为了通过测试直接放大容差。
- NaN/Inf、signed zero、Cautious mask 和 AMSGrad 选择规则另做逐位或分类断言，不能只看近似值。
- `Reference` 永远保留原实现，供需要当前逐步物化行为的用户显式选择。

### 2.5 dtype、超额精度链与状态边界

P1/P2/P3 的专用路径首版只开放 **F32 非量化输入**，并要求 `param/grad/m1/m2/max_m2` 的 shape、device、dtype 一致。F16、BF16、F64、Flex32、量化、不同 state dtype、空张量都先回退 Reference；P3 的向量化和阈值调优不得顺带扩大 dtype 范围。

半精度不能仅靠误差容差开放。锁定 CubeCL 的 `F16Evaluation::Chain` 会把一串 F16 算术保持在 F32 中，`Accumulators` 还会扩大保持范围，普通 immutable `let` 并不结束链；而 Burn Reference 的 `grad.square()`、`mul_scalar`、`add` 等每个 tensor op 都有独立的 dtype 物化边界。例如 F16 首步 `grad=300, beta_2=0.999`：Reference 先把 `300*300=90000` 物化为 F16 `Inf`，再得到 `m2=Inf`；若 fused 将 `(g*g)*factor_2` 整段保持在 F32，则会得到有限的 `m2≈90`。这会同时改变 state 和参数更新，不能归入 FMA 或 `4ε/8ε` 误差。

首版因此对半精度统一回退，不根据机器猜测 evaluation mode。未来若要开放 F16/BF16，必须作为独立变更在每个参考 tensor-op 边界插入由目标 CubeCL API **保证**的 dtype 舍入/物化，至少覆盖 square、每次矩更新乘加、bias correction、sqrt/add/div、decay 和最终参数更新；仅拆成另一个 immutable `let` 无效。还必须分别用 `PerOperation`、`Chain`、`Accumulators` 配置验证上述 `grad=300` 溢出用例及完整特殊值矩阵。无法表达或证明这些边界时继续回退，不能放宽容差通过。

不把 F16 参数加 F32 动量称为“master weights”。真正的 master weights 需要 FP32 参数副本、初始化、checkpoint 迁移和回写规则，不属于本方案。现有首步 state 继续从 grad 派生并保持原 dtype。

## 3. 主流框架实现对照

以下结论来自源码，不把其他框架的默认值当作 Burn 的兼容目标。

| 框架 | 已核实的实现 | 对本方案的启发 | 明确不照搬的部分 |
|---|---|---|---|
| PyTorch `torch.optim.AdamW` | 有 single-tensor、foreach、fused 三条路径；只有未指定时才自动选择，自动路径不会强制 fused。按 `(device, dtype)` 分组处理 tensor lists。 | 保留参考路径； fused 与 foreach 分开交付；分组必须考虑 device/dtype。 | Burn 的 epsilon、默认值、cautious、逐参数 lr/time 不同，不能只按 `(device,dtype)` 分组。 |
| PyTorch CUDA `MultiTensorApply.cuh` | `kChunkSize=65536`、`kBlockSize=512`；每个 tensor 保留独立地址，以 `block_to_tensor` 和 `block_to_chunk` 映射工作；受每次 launch 的 tensor/block metadata 容量限制，必要时多次 launch。 | 不做全模型物理拼接；按 chunk 工作；必须有 launch 上限和拆分规则。 | Burn 的 `Sequence` 是编译期展开，不应假装具有 PyTorch 的运行时 metadata 索引。常量也不能未测量就复制。 |
| Optax `adamw` | `scale_by_adam -> add_decayed_weights -> scale_by_learning_rate`；state 是 pytree，支持 `mu_dtype`。 | 保持变换职责清晰，decay 不进入动量。 | 其组合式和 XLA 编译模型不要求 Burn 采用同一抽象；本方案不改变 state dtype。 |
| Keras 3 Adam/AdamW | 复用后端 tensor ops，moment/update 在 Adam 中，weight decay 由基类机制处理；没有据此推断必然 fused。 | 后端无关参考实现仍是正确性的锚点。 | 不使用 Keras 的 `alpha=lr*sqrt(1-beta_2^t)/(1-beta_1^t)` 改写 Burn 的舍入顺序。 |

参考资料见[第 15 节](#15-参考资料)。

## 4. 分阶段交付与门禁

| 阶段 | 内容 | 可独立交付的价值 | 合并门禁 |
|---|---|---|---|
| P0 | 固定 reference helper、测试和 benchmark | 建立可比较基线 | 测试证明参考路径行为未变 |
| P1 | 单参数 out-of-place CubeCL Fused | 消除中间 tensor 和大部分逻辑启动 | 数值、布局、decorator、checkpoint、benchmark 全部通过 |
| P1b | 可选 COW 输出复用 | 降低稳态分配与回收 | alias/COW 回归通过，不能快于正确性门禁而合并 |
| P2 | 模块级 Foreach 收集、分组、多参数内核 | 减少多参数启动次数 | P1 已稳定；两次 module map、分组、chunk 上限均有测试和实测收益 |
| P3 | 可选向量化与阈值调优 | 提高大参数吞吐 | 不对齐/尾部/非连续布局测试；不扩大 dtype 范围 |

P2 不阻塞 P1。若 P2 的两次 module traversal 或 JIT 展开成本抵消收益，停止 P2，不影响已交付的 P1。

## 5. P0：可复现基线

### 5.1 固定参考实现

把当前 `AdamW::step` 和 `AdaptiveMomentumW::transform` 的原函数体整理为私有 `reference_step`，先运行现有测试，再接入任何新分派。这样：

- `Reference` 调用同一份原实现，避免维护两套数学公式；
- Fused 测试在同一 device 上比较 reference 与 candidate；
- 默认配置完全不走新代码；
- 后续差异可以归因于内核，而不是重构时顺手改公式。

### 5.2 新增执行方式，但不改变旧配置

建议增加一个枚举，避免 `fused=false, foreach=true` 这种无定义组合：

```rust
#[derive(Config, Debug)]
pub enum AdamWExecutionConfig {
    /// 当前逐步 tensor 实现；默认值。
    Reference,
    /// 每个参数一次融合内核；不支持时回退 Reference。
    Fused,
    /// 优先多参数融合；只有单项时用 Fused；不支持时回退 Reference。
    Foreach,
}
```

`AdamWConfig` 只新增 `execution: Option<AdamWExecutionConfig>`，不再额外依赖 `#[config(default)]` 做迁移：上一版本 JSON/binary 缺少该字段时，Serde 对 `Option` 得到 `None`，`build()` 将 `None` 解析为 `Reference`；`with_execution(...)` 写入 `Some(...)`。因此 `AdamWConfig::new()` 与旧配置反序列化的**语义默认值**都是 Reference，但不要求配置字段本身是非 Option 的 `Reference`。`AdamW` 保存解析后的非 Option execution。直接调用 `Optimizer::step` 时，`Foreach` 等价于 `Fused`，因为单参数接口没有批次上下文。

必须增加上一版本配置（不含 `execution`）的 JSON 与 `load_binary` 回归测试，证明加载后走 Reference；再验证显式 Fused/Foreach 的 round-trip。不能把只影响 `new()` 的 `#[config(default)]` 当成反序列化迁移机制。原有字段、默认值、builder 和 clipping 全部保留。

这是新增 public API，实施 PR 必须同时更新 rustdoc 和 `burn-book/src/changelog.md`，说明默认行为没有改变以及 `None` 的兼容语义。

### 5.3 Benchmark 先分清测量对象

新增 `burn-optim` 自有 optimizer-step benchmark，至少报告：

1. Host enqueue 时间，不在计时区 sync；
2. enqueue + `Device::sync` 的总时间；
3. 首次 JIT/编译时间，与稳态分开；
4. step 作用域结束后的 drop/sync 时间；
5. 参数总元素数、总字节数、参数个数和 launch 数（可获得时）。

参数规模建议：

| 模型形态 | 参数张量 |
|---|---|
| 微小模型 | 30 个 `[64,64]`，F32 |
| 混合 rank | bias `[64]`、linear `[64,64]`、conv `[16,8,3,3]` |
| 大参数 | 1、4、16 个 `[1024,1024]` |
| 长短混合 | 多个小 bias 与 1～4 个大矩阵混合 |

每组至少覆盖 F32；F16/BF16 只验证 Reference 回退并记录基线，不纳入首版专用路径的加速验收。记录硬件、OS、Rust/Cargo feature、fusion on/off、编译模式、样本数、warmup、median/p95 和原始结果。Batch size 不进入纯 optimizer-step benchmark；它只影响前反向，不改变已生成梯度的 shape。

## 6. P1：单参数 Fused 架构

### 6.1 文件与 feature 组织

建议变更：

```text
crates/burn-cubecl/src/kernel/optim/mod.rs       # 新模块
crates/burn-cubecl/src/kernel/optim/adamw.rs     # launch、kernel、输出规划
crates/burn-optim/src/optim/adamw.rs             # reference + execution 分派
crates/burn-optim/src/optim/adamw/cube.rs        # 高层 Tensor 桥接
crates/burn-optim/src/optim/mod.rs               # 按现有规则重导出 execution 配置类型
crates/burn-optim/Cargo.toml                     # 可选 burn-cubecl、runtime features 与 bench target
crates/burn-optim/benches/adamw.rs               # Divan 计时与条件编译的 path-verification
crates/burn/Cargo.toml                           # 向 burn-optim 转发已有 runtime features
```

`burn-optim` 增加可选 `cubecl-backend = ["dep:burn-cubecl"]`，`burn-cubecl` 依赖必须设 `default-features = false`（其默认 feature 包含 Fusion，否则 raw 快路径会被意外关闭），并由现有 `std` feature 追加 `burn-cubecl?/std`。同时增加与其他扩展 crate 一致的 `cpu/cuda/rocm/wgpu/...` 转发 feature；其中 `rocm` 映射到 `burn-cubecl/hip`，不是不存在的 `burn-cubecl/rocm`。另定义 `fusion = ["burn-core/fusion", "burn-cubecl?/fusion"]` 专门验证回退路径；这里必须使用 weak 转发，不能在 Flex+Fusion、未选择任何 Cube runtime 的合法组合中激活要求 runtime 的 `burn-cubecl` 可选依赖。观测 feature 必须定义在实际提交 launch 的 crate，并由上层转发，不能只在 `burn-optim` 定义空 feature：

```toml
# crates/burn-cubecl/Cargo.toml
adamw-bench-observability = []

# crates/burn-optim/Cargo.toml
adamw-bench-observability = ["burn-cubecl?/adamw-bench-observability"]
```

该 feature 只编译第 11.1 节的观测接口，不改变分派，也不向正常用户构建暴露。`burn-cubecl` 内部的 `cfg(test)` 只覆盖它自己的单元测试；作为 `burn-optim` 普通依赖编译时不会继承上层的 `cfg(test)`，所以 burn-optim 的路径验证测试和独立 bench 都必须显式启用这个转发 feature。未启用 CubeCL 时仍是原来的 `no_std`/Flex 实现，不引入 CubeCL 依赖。`burn` 顶层的同名 runtime feature 转发给 `burn-optim?`；直接使用 `burn-optim` 的用户需显式启用对应 feature。benchmark target 还需在 dev-dependencies 中显式加入仓库现有版本的 `divan`，并禁用默认 libtest harness：

```toml
# crates/burn-optim/Cargo.toml
[[bench]]
name = "adamw"
path = "benches/adamw.rs"
harness = false
```

计时入口必须调用 `divan::main()`，不能只添加 `benches/adamw.rs` 后依赖 Cargo 自动发现。`--verify-path` 在进入 Divan 前由 target 自行解析并走独立验证入口；正常计时参数仍交给 Divan。

raw 测试/benchmark 必须同时关闭整条默认 feature 链，而不是只关闭直接 `burn-cubecl` 的默认 feature：`burn-optim` 使用 `--no-default-features --features std,cpu`，避免 `burn-core/default -> burn-cpu?/default -> burn-cubecl/default -> fusion` 被重新打开。`burn-nn` dev-dependency 当前显式写了 `features=["default"]`，也必须改成 `default-features=false, features=["std"]`，否则即使命令行使用 `--no-default-features`，测试依赖仍会把 `burn-core/default` 加回来。Fusion 回退测试另用 `--no-default-features --features std,cpu,fusion`，不与 raw 配置共用结论。

不修改 `burn-backend::Backend`，因此不会要求 Flex、Remote、自定义 backend 增加 AdamW 方法，也不会产生 `unimplemented!()` 默认实现。

### 6.2 高层分派

伪代码只表达控制流，实施时以当前 Rust/CubeCL API 为准：

```rust
match self.execution {
    Reference => self.reference_step(lr, tensor, grad, state),
    Fused | Foreach => {
        match cube::try_step(lr, tensor, grad, state, &self.momentum, self.weight_decay,
                             self.cautious_weight_decay) {
            Ok(updated) => updated,
            Err(original_inputs) => self.reference_step_with(original_inputs),
        }
    }
}
```

`try_step` 必须先检查 execution、dtype、shape、device 和空张量，再尝试 primitive 转换。不得在内核已经部分执行后“回退”。

为处理 `Cube = CubeBackend` 与 `Cube = Fusion<CubeBackend>` 两种编译形态，`burn-cubecl` 提供一个小型 `try_adamw` 边界：

- `not(feature="fusion")`：接收 raw `CubeTensor`，执行专用内核；
- `feature="fusion"`：原样返回所有 primitive，让 `burn-optim` 走 reference，继续由现有 Fusion 处理原算子。

这样不让 `burn-optim` 猜测 Cube 的 feature 形态，也不伪造 Fusion IR。若未来要让专用内核进入 Fusion，必须另提 `CustomOpIr`，声明全部读写、输出 shape/dtype、alias、graph capture 和 Remote handler，不在 P1 隐式支持。

### 6.3 转换失败必须保留原输入

`try_into_primitive` 会消费 tensor，而其错误不携带原 tensor。桥接采用两阶段转换：

1. 先用 device/kind/dtype 检查排除明显不支持的输入；
2. 对仍保留的原始 `Tensor` 做句柄级 `clone`，用 clone 尝试转换；
3. 所有 param/grad/state 都成功后才 drop 原始句柄并进入 launch；
4. 任一失败则 drop 已转换的 clone，原始输入仍完整，调用 reference；
5. `burn-cubecl::try_adamw` 在 Fusion 形态返回原 primitive，桥接恢复 `Tensor` 后走 reference。

clone 只增加临时句柄所有权，不复制 tensor 数据。进入 COW 判断前必须 drop 这些临时句柄，否则会错误地把可复用输入判为共享。

### 6.4 输出与 checkpoint 恢复

内核输出逻辑上是：

```rust
struct AdamWOutput {
    param: CubeTensor,
    moment_1: CubeTensor,
    moment_2: CubeTensor,
    max_moment_2: Option<CubeTensor>,
}
```

所有输出保留输入的 shape、strides 语义、dtype 和 device；rank 从未被擦除为永久 1D。桥接在原 `D` 下构造 `Tensor<D>` 和 `AdamWState<D>`，Host 只推进原有 `state.time`。因此现有 `RecordState`、`DynState::create(..., D)`、Burnpack 名称和 `__rank` 不变。

## 7. P1：内核与 launch 细节

### 7.1 使用当前 LinearView API

实现采用与 `unary_float.rs`、`binary.rs` 相同的机制：

- `LinearView` / `LinearViewMut`；
- `output.is_in_bounds(ABSOLUTE_POS)`，越界 `terminate!()`；
- `calculate_cube_count_elemwise` 和合适的 `CubeDim`；
- `address_type = "dynamic"`，Host 从全部输入/输出求地址类型；
- `dtype_to_storage_type`；
- P1 使用 `Const<1>`，不在首版引入对齐向量尾部的复杂度。

不能再使用原稿中不受当前 API 支持的 `Tensor::is_in_bounds/read/write` 组合。

### 7.2 Launch 参数

Host 参数与 launch 参数必须分开：

```rust
// 仅存在于 Host，保持参考实现的计算精度。
struct AdamWHostScalars {
    lr: f64,
    beta_1: f32,
    beta_2: f32,
    factor_1: f32,
    factor_2: f32,
    bias_1: f32,
    bias_2: f32,
    epsilon: f32,
    decay_rate: f64,
    one_minus_decay_rate: f64,
}

// launch 前按本组 tensor storage dtype 转换；这里不是 F64 参数。
struct AdamWLaunchScalars {
    lr: InputScalar,
    beta_1: InputScalar,
    beta_2: InputScalar,
    factor_1: InputScalar,
    factor_2: InputScalar,
    bias_1: InputScalar,
    bias_2: InputScalar,
    epsilon: InputScalar,
    decay_rate: InputScalar,
    one_minus_decay_rate: InputScalar,
}

// 进入 JIT key 的结构变体。
enum DecayMode { None, Standard, Cautious }
struct AdamWVariant {
    has_state: bool,
    has_old_max: bool,
    amsgrad: bool,
    decay: DecayMode,
}
```

`bias_1/bias_2` 是分母，不是倒数。`has_state/has_old_max/amsgrad/decay` 是 compile-time variant；数值是 runtime scalar，避免学习率 schedule 每步触发 JIT。所有 `InputScalar` 都按 tensor dtype 创建，尤其是 Host 为 f64 的 `lr/decay_rate/one_minus_decay_rate`：首版 F32 kernel 不得因为标量注册而引入 F64 类型或 `SHADER_F64` 要求。只有未来开放 F64 tensor 时才使用 F64 scalar。参考 `burn-cubecl/src/ops/tensor.rs` 的现有 `InputScalar` 转换方式，增加无 `SHADER_F64` 的 F32 compile+run 门禁。

无旧 state 或无 AMSGrad 时，不为可选输入读取未初始化内存。`amsgrad && !has_old_max` 使用新 `m2` 初始化；`!amsgrad && has_old_max` 仅在 Host 原样携带旧 max，不在内核读写。可以使用独立 kernel variant，或给未使用输入传合法 dummy view 并保证对应分支在 compile-time 被移除；实施时优先选择不会让 alias planner 误认所有权的独立 variant。

### 7.3 与参考顺序一致的内核骨架

以下是与当前 CubeCL 风格对齐的**设计伪代码**，不是声称已经编译验证的源码：

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn adamw_kernel<F: Float>(
    param: LinearView<'_, F>,
    grad: LinearView<'_, F>,
    old_m1: LinearView<'_, F>,
    old_m2: LinearView<'_, F>,
    old_max: LinearView<'_, F>,
    mut out_param: LinearViewMut<'_, F>,
    mut out_m1: LinearViewMut<'_, F>,
    mut out_m2: LinearViewMut<'_, F>,
    mut out_max: LinearViewMut<'_, F>,
    scalars: &AdamWScalarsLaunch,
    #[comptime] variant: AdamWVariant,
    #[define(F)] _dtype: ElemType,
) {
    if !out_param.is_in_bounds(ABSOLUTE_POS) { terminate!(); }
    let i = ABSOLUTE_POS;
    let p = param.read(i);
    let g = grad.read(i);

    let beta_1 = scalars.beta_1.get::<F>();
    let beta_2 = scalars.beta_2.get::<F>();
    let factor_1 = scalars.factor_1.get::<F>();
    let factor_2 = scalars.factor_2.get::<F>();
    let m1 = if variant.has_state {
        old_m1.read(i) * beta_1 + g * factor_1
    } else {
        g * factor_1
    };
    let m2 = if variant.has_state {
        old_m2.read(i) * beta_2 + (g * g) * factor_2
    } else {
        (g * g) * factor_2
    };

    let v = if variant.amsgrad {
        let previous = if variant.has_old_max { old_max.read(i) } else { m2 };
        select(previous < m2, m2, previous)
    } else { m2 };

    let delta = (m1 / scalars.bias_1.get::<F>())
        / ((v / scalars.bias_2.get::<F>()).sqrt() + scalars.epsilon.get::<F>());

    let decayed = match comptime![variant.decay] {
        DecayMode::None => p,
        DecayMode::Standard => p * scalars.one_minus_decay_rate.get::<F>(),
        DecayMode::Cautious => {
            let same_sign = (p >= F::new(0.0)) == (m1 >= F::new(0.0));
            let raw = p * scalars.decay_rate.get::<F>();
            p - select(same_sign, raw, F::new(0.0))
        }
    };

    out_m1.write(i, m1);
    out_m2.write(i, m2);
    if variant.amsgrad { out_max.write(i, v); }
    out_param.write(i, decayed - delta * scalars.lr.get::<F>());
}
```

实施时必须确认 scalar 的 LaunchArg 转换、`InputScalar::get` 的实际返回形式、`select` 的标量形式和 comptime match 的实际宏语法；不能直接把伪代码当作生产代码提交。`!amsgrad` 时的 max 输出不是计算结果：Host 必须把旧 `Option<max_moment_2>` 原样放回 `AdamWState`，且 delta 仍使用 `m2`；相应 kernel variant 可以完全不接收 `old_max/out_max`。

### 7.4 空张量、非连续布局和 tiled 输入

- 空张量在 Host 回退 reference，保持 time 推进和 state 初始化语义，避免 `working_units=0` 的 launch 特例。
- 非连续但非重叠输入通过 `LinearView` 读取，不强制 `reshape` 或 `into_contiguous`。
- 输出优先使用连续、同 shape 的新 tensor；若未来要复用非连续输入，必须确认输出 view 的逻辑线性映射与所有输入一致。
- storage-tiled 输入在 P1 回退 reference，不把 tile strides 当普通行布局。
- rank 0、含 1 维、非标准 stride 都加入测试；跨 rank 不改变数值顺序。

### 7.5 P1 先 out-of-place，P1b 再复用

P1 首版为 param/m1/m2 以及需要更新的 AMSGrad max 分配独立输出。这样 param、grad 和 state 即使共享存储，内核也只读输入，不会在中途覆盖后续读取。非 AMSGrad 的旧 max 不更新、不分配新副本，按参考实现原样保留。

P1b 对每个输出独立判断：

1. 输入 `can_mut()` 且 `is_nonoverlapping()`，不是 tiled；
2. 输入 allocation id 不与任何其他角色的 allocation id 相同；
3. 输入与输出的 shape/dtype/layout 相容；
4. 所有临时 clone 已 drop；
5. alias 参数的 input position 与实际 launch 签名一致。

任一条件失败就分配输出。`as_linear_view_alias` 只在满足全部条件时使用。分配标识可比较 `handle.service` 与 `handle.memory.descriptor().id`；同一 allocation 的不同 slice 也先保守视为 alias，未来有区间证明再放宽。

AMSGrad 首步以及已有 state 但 `max_moment_2=None` 的情况都没有 old max，不得复用 dummy 输入；此时为新 max 分配输出。非 AMSGrad 保留旧 max 的路径不参与 max COW 写入。P1b 需要分别覆盖 param 共享、state 共享、param==grad、m1==m2、max 与 m2 重叠等情况。

## 8. Clipping、Autodiff、Decorator 与 Checkpoint

### 8.1 Clipping 保持在融合之前

P1 继续由 `ModuleOptimizerMapper` 先调用 `clip_gradient`，再把 clipped grad 交给 fused kernel。P2 在收集每个参数时执行同样的逐参数 clipping，然后才能分组。

**禁止**把 clipping 系数折进 `lr`。若 `g' = alpha*g`，正确顺序是先用 `g'` 更新 m1/m2，再计算 delta；weight decay 仍只依赖原参数和原 lr。简单使用 `lr*alpha` 会同时写错 state 和参数。

 clipping on/off 分开 benchmark，因为 clipping 自身的 norm/reduction 可能主导启动数。

### 8.2 执行模式矩阵

| 模式 | P1/P2 行为 |
|---|---|
| Reference | 原实现，不检查 Cube 快路径 |
| raw Cube eager | 合格输入进入专用内核 |
| Autodiff device | 沿用 `tensor.inner()` 和 `ParameterContext::restore`，优化器更新本身不建反向节点 |
| Fusion | `try_adamw` 返回原输入，走 reference；不伪装支持 CustomOpIr |
| Remote | primitive 不匹配，走现有 reference 远程算子 |
| Capture backend | primitive 不匹配，走 reference；普通 raw Cube graph capture 另测 launch 可记录性 |
| Group | primitive 不匹配，走 reference |
| 量化/不支持 dtype/layout | 回退 reference |
| `step_multi` | 先沿用 GradAdaptor 的梯度合并/device 结果，再按第 9.4 节捕获的有效执行 stream 分组和提交 |

回退只发生在启动前。Kernel launch 失败不能悄悄换算法再执行一次，否则可能重复更新。

### 8.3 Checkpoint 验收

格式不变需要通过行为证明，而不是仅看结构体：

1. Reference 跑 N 步并保存，Fused 加载后跑第 N+1 步，与同起点 Reference 比较；
2. Fused 跑 N 步并保存，Reference 加载后继续，反向覆盖；
3. Foreach 重复以上两种方向；
4. AMSGrad 开关分别覆盖，并显式测试“无 max 的旧 state→AMSGrad”和“含 max 的旧 state→非 AMSGrad”两条 checkpoint 迁移路径；
5. 检查每个 `ParamId` 的 `__rank`、time、tensor 名称、shape、dtype；
6. 加载后 group/path 仍按原规则匹配，逐参数 lr 不变；
7. 对 state 值使用本方案的 dtype 容差，不要求不同执行方式的浮点结果必然 bitwise 相同。

## 9. P2：模块级 Foreach 数据流

### 9.1 为什么需要两次 module map

当前 `ModuleMapper::map_float` 必须立即返回一个参数，而 raw Cube kernel launch 在调用时就已排队，无法先返回参数再收集“未来的 batch”。因此不能只在 `AdamW::step` 内放一个 `Vec` 就实现全模型 foreach。

P2 使用以下流程：

```text
module.map(CollectMapper)
    ├─ 无梯度：原样返回，不推进 time
    ├─ 非 AdamW/非 Foreach/不合格：立即执行原 step_dyn
    ├─ rank 不在 dispatch_rank! 支持表：立即执行原 step_dyn，不生成 PendingAdamW
    └─ 合格 AdamW：保留原参数，移除梯度，生成 PendingAdamW
                  ↓
        partition + chunk + launch
                  ↓
module.map(ApplyBatchMapper)
    └─ 按 ParamId 替换更新后的参数，恢复 ParameterContext 和 ParamMapper
```

两次遍历只做 host 元数据操作，不做 tensor 打包或 D2H 同步。其成本必须计入 benchmark。

rank 准入必须先于 primitive 转换、梯度移除和 `StateBuilder` 创建。当前 `dispatch_rank!` 只支持 0–8；`D > 8` 在收集阶段调用原 `step_dyn`，由同一支持表按当前行为立即 panic。此时 partition/chunk/launch 尚未开始，所以不会产生任何 Foreach 参数更新或专用 state。不得改用泛型 `reference_step` 让 rank 9 首次更新成功，再把失败推迟到下一次 `to_device_dyn`、`state_flatten` 或 `state_unflatten`。若未来正式支持更高 rank，必须同时扩展 `step_dyn/to_device_dyn/state_flatten/state_unflatten` 的所有动态路径，而不是只放宽 Foreach。

### 9.2 发现具体 AdamW

`ModuleOptimizer` 存储的是 `Arc<dyn DynOptimizer>`。可以增加一个默认返回 `None` 的 hidden capability 方法：

```rust
#[doc(hidden)]
fn as_any(&self) -> Option<&dyn core::any::Any> { None }
```

现有 blanket `impl<O: Optimizer> DynOptimizer for O` 返回 `Some(self)`。自定义 `DynOptimizer` 不受影响，默认没有 foreach 能力。收集器只在 `downcast_ref::<AdamW>()` 成功且 execution 为 `Foreach` 时建立 pending。

这不是 backend capability 查询；backend 是否可执行仍由第 6 节的 primitive 边界决定。

### 9.3 Pending 必须保存的信息

```text
PendingAdamW
├─ ParamId 与用于写回 state 的原 map key
├─ optimizer 的完整数值配置和 grad_clipping 后梯度
├─ lr: f64
├─ rank（仅用于恢复，不参与线性寻址）
├─ param/grad/m1/m2/max_m2 primitives
├─ has_state、has_old_max、原 state.time、更新后 time 和 bias 分母
├─ device、client service、有效执行 stream、dtype、numel、layout/address 信息
├─ optimizer Arc、path、grad_clipping（恢复 OptimizationContext）
└─ StateBuilder
```

`StateBuilder` 在泛型 `map_float<const D: usize>` 中、通过上述动态 rank 准入后才创建，可表示为一个 `FnOnce(RawAdamWState) -> DynState`：它把返回的 primitive 重新构造成 `AdamWState<D>` 并调用 `DynState::create(..., D)`。这样无需把不同 rank 的 state 塞进不存在的 `Vec<AdamWState<D>>`，也不丢失 `__rank`，更不会绕过 `dispatch_rank!` 的 0–8 限制。

`ApplyBatchMapper` 再次 `Param::consume`，按 `ParamId` 取输出，调用 `ParameterContext::capture/restore` 和 `Param::from_mapped_value`。没有输出的参数原样返回。

### 9.4 捕获并恢复有效执行 stream

Pending 中的 stream 是**收集时这次原 `step_dyn` 本应使用的有效执行 stream**，不是 parameter、gradient 或 state allocation 的创建 stream。对 unpinned client，它来自当时的 `StreamId::current()`；对显式 pinned client，它来自 client 固定的 stream。`Handle.stream` 只表示 handle 创建位置，不能用于这个判断。

当前锁定 CubeCL 的 `ComputeClient::stream_id()` 已正确实现 `Some(pinned) -> pinned / None -> StreamId::current()`，但方法是私有的。P2 实施前必须在指定 CubeCL revision 提供等价只读 API（或复用升级后已有 API），例如：

```rust
pub fn execution_stream(&self) -> StreamId {
    self.stream_id()
}
```

`burn-cubecl` 的 P2 边界在 CollectMapper 仍在原执行作用域内时查询并保存该值，同时保存 `client.service_id()`。若使用的 revision 没有可查询 pinned/unpinned 有效 stream 的能力，整个 P2 不得启用，`Foreach` 保持 Reference；不能假定 client 未 pinned，也不能改用 `Handle.stream`。P1/P1b 在原调用点立即 launch，不依赖这个延迟执行前置条件。

分组和 chunk metadata 可以在外层计算，但每组的**全部输出分配、单项 P1、所有 chunk launch，以及收集后才确定的 Reference 回退**必须整体置于保存的有效 stream 中：

```rust
for group in groups {
    group.execution_stream.executes(|| {
        // 使用 service_id 相同的代表 client 分配该组全部输出；
        // 在该作用域内提交单项 P1 或本组每个 chunk launch。
        // 需要 Reference 的 delayed item 也在其各自保存的 stream 中执行。
    });
}
```

这样即使外层默认 stream 已从 A 切换到 B，A 上收集的参数仍在 A 更新，A 上后续消费者保持原有排队顺序，输出 allocation 也归属正确的 stream。观测 snapshot 需同时记录实际 launch 的 service/stream，测试不能只检查分组 key。

### 9.5 分组键，不按大小粗暴分两桶

第一版按以下完整 key 分组：

```text
(concrete Cube device, client service_id, effective execution stream id, dtype,
 lr.to_bits(),
 state.time / time_i32,
 beta_1.to_bits(), beta_2.to_bits(), epsilon.to_bits(),
 weight_decay.to_bits(),
 has_state, has_old_max, amsgrad, decay_mode)
```

原因：

- 不同 device、client service 或有效执行 stream 不能为了少一次 launch 改变顺序；
- `ModuleLearningRate` 可以逐参数不同；
- 缺梯度或加载历史会让 time 不同；
- NaN 超参数用 bits 分组，避免把 `NaN != NaN` 变成异常分派；
- `has_state`、`has_old_max`、AMSGrad 和 decay 是不同 kernel variant；已有 state 但无 old max 不能与已有 old max 共用读取路径，非 AMSGrad 保留旧 max 也不能被错误清掉。

在当前动态支持的 rank 0–8 内，rank 和 shape 不必相同；非连续 layout 由每个 view 自己的 metadata 描述。首版只支持同 dtype 组。后续若引入 per-tensor scalar list，可取消 lr/time 分组，但必须先验证 metadata 上传成本，不能假设一定更快。

### 9.6 不合格项和失败边界

- 空组直接返回；
- 只有一项时复用 P1 单参数 fused，但仍在第 9.4 节保存的有效 stream 中分配输出并 launch；
- 收集时已发现不满足 P2 条件的项立即走原 `step_dyn`；收集后才确定的 P1 或 Reference 回退必须在该项保存的有效 stream 中执行，不能在外层 stream 静默换算法；
- 分组、chunking 可以在外层准备，输出分配必须在对应 stream 作用域内、launch 前完成；
- P2 初版所有输出 out-of-place，因此不同参数或 state 共享 allocation 时也不会写坏输入；
- P2 的 COW 复用以后只复用 P1b 已验证的 planner，不另写一套所有权规则。

## 10. P2：多参数内核的可实现设计

### 10.1 不做矩形 `MaxSmallLen × N` 网格

简单以最大 tensor 长度发射会让短 tensor 的线程空转。首版采用**一维 chunk 列表**：

```text
host:
    cube_dim = 一维，N = 1
    chunk_len = cube_dim.x
    对每个 tensor：
        chunks = ceil(numel / chunk_len)
        为每个 chunk 记录 tensor_index、local_offset、valid_len
    cube_count.x = 本批全部 chunk 数
```

每个 chunk 对应一个 cube，线程局部位置为 `local = UNIT_POS_X`，真正的 tensor 逻辑索引统一为 `index = local_offset + local`。只处理 `local < valid_len` 的元素；所有 param/grad/state read 和所有 output write 都必须使用同一个 `index`，`LinearView` 不会自动注入 chunk offset。大 tensor 自然占多个 chunk，小 tensor 只占一个，不需要固定 65,536 阈值，也不需要把长短 tensor 分成两个算法。

### 10.2 使用 `Sequence` 的实际限制

CubeCL 的 `Sequence<T: LaunchArg>` 可以注册动态数量的参数，但其索引和迭代在**编译期展开**。Burn Fusion 的 `GlobalArgs { tensors: Sequence<GlobalTensor>, ... }` 已证明这一参数组织方式可用，但不能像 PyTorch 一样在内核里用运行时 `tensor_id` 直接索引。

P2 因此使用编译期 chunk descriptor 和 uniform 分支：

```rust
#[unroll]
for chunk in 0..args.chunks.len() {
    if CUBE_POS_X == chunk {
        let t = comptime![args.chunks.index(chunk).tensor_index];
        let offset = comptime![args.chunks.index(chunk).local_offset];
        let local = UNIT_POS_X;
        if local < comptime![args.chunks.index(chunk).valid_len] {
            let index = offset + local;
            // 对 args.param[t]、args.grad[t]、args.m1[t]... 的所有 read，
            // 以及 param/m1/m2/max 的所有 write，统一使用 index。
        }
    }
}
```

每个 cube 的 `CUBE_POS_X` 在 cube 内一致，不会产生同 cube 内按元素搜索 tensor 的发散。实际实现需要把 `Sequence<LinearView...>` 的 launch 注册和 comptime 索引写成当前宏接受的形式，并先用 2～3 个 tensor 做 compile-only proof。

### 10.3 Launch 上限与 JIT 成本

每次 launch 的参数数量和展开后的代码随 tensor/chunk 数增长，所以必须有上限：

- `max_tensors_per_launch`：先取保守值并跨 runtime 验证；
- `max_chunks_per_launch`：限制 JIT 代码量和 launch 参数；
- `max_unrolled_elements_metadata`：必要时进一步限制 Host descriptor 大小。

常量不能照抄 PyTorch 的 110/320 或 65,536。先以较小值（例如 32 tensors、256 chunks）完成正确性，再依据 CUDA/ROCm/Metal/WGPU/CPU 的参数限制和编译耗时调优。超过任一上限就拆分下一次 launch；没有全模型固定 launch 数承诺。

JIT key 包含 dtype、variant 和 chunk descriptor；模型 shape 稳定时后续 step 可复用编译结果。Benchmark 必须分别报告首次编译和稳态，防止把每步重新 JIT 当成融合收益。

### 10.4 P2 的 traffic 与 launch 估算

对每个无 AMSGrad 的已初始化参数，Fused 的逻辑数据流是：

```text
read:  param, grad, old_m1, old_m2       = 4 个 tensor streams
write: new_param, new_m1, new_m2         = 3 个 tensor streams
```

Foreach 不进一步降低这个每元素 traffic，只降低 launch/host 开销。AMSGrad 在 `has_old_max=true` 时增加 old max read，并写 new max；无 old max 时只写新 max。首步不读旧 state；P1b 是否原地复用不会改变逻辑读写数量，但会改变分配与回收成本。

原稿的“显存往返降低 73.3%”“瞬态分配为 0”全部删除。实际报告只使用 profiler 的 bytes、allocation 和 launch 数据。

## 11. 正确性测试矩阵

### 11.1 数值与状态

**先证明走了目标路径，再比较数值。** raw 测试首先断言 `!burn_cubecl::FUSED`。计数器和 snapshot/reset 实现位于实际提交专用 launch/chunk 的 `burn-cubecl`，只在 `#[cfg(any(test, feature = "adamw-bench-observability"))]` 下编译，并提供 `#[doc(hidden)] pub` 接口；`burn-optim` 的同名 feature 按第 6.1 节转发到 `burn-cubecl?/adamw-bench-observability`。因此 burn-optim 的路径验证测试、Fusion 回退计数测试和独立 benchmark 都要显式启用 `adamw-bench-observability`；普通 `std,cpu` 构建只验证不依赖观测接口的功能，未启用该非默认 feature 的生产库不包含此接口。只有 raw `try_adamw` 真正提交专用内核后才计数，参考回退不计数。

所有**会触发专用路径**的测试（不只是读取计数的测试）必须共用一把测试锁；在锁的临界区内依次取基线、执行、`Device::sync`、取差值并检查输出。`AtomicUsize` 只解决数据竞争，锁才消除 before/after 之间的测试串扰。单参数 Fused 断言计数增加；Foreach 断言批次数和 chunk 拆分次数；Fusion 配置断言 `burn_cubecl::FUSED` 且专用计数不增加。没有观测 feature 的测试不得编译或引用这些计数接口。

独立 `--bench adamw` 分两步：先用 `adamw-bench-observability` 构建运行一次 path-verification，打印 execution、`FUSED` 和实际专用 launch/chunk 数，并在共享锁内隔离各 case；再用相同 raw `std,cpu`、但不启用观测 feature 的构建采集性能数据，避免把计数器开销和串行化 instrumentation 计入稳态结果。bench target 内 verification 的 import、helper 和分派函数必须整体放在 `#[cfg(feature = "adamw-bench-observability")]` 下；无 feature 的同名分派函数只能在 `--verify-path` 时报告“请使用观测 feature 重建”并以失败状态退出，不能引用 snapshot/reset 或 verification helper。这样第二条命令不会类型检查已移除的接口。

`--verify-path` 在调用 `divan::main()` 前由 target 自己解析；验证入口维护预期 case ID 清单，每实际完成一个 case 就记录一次，最后断言完成集合与预期完全相同且非空。参数解析错误、过滤成 0 case 或只启动 runner 而没有执行验证体都必须失败，不能打印 counters 后就宣称路径通过。正常计时路径由 `divan::main()` 启动，且 target 必须按第 6.1 节设置 `harness = false`。禁止只凭输出一致就声称测到了新内核。

| 类别 | 必须有区分度的用例 |
|---|---|
| 基础 | 首步、第二 step、50 steps；非对称正负梯度；参数/m1/m2/max 全部比较 |
| decay | `weight_decay=0` 且 `p=Inf`，确认不引入 `0*Inf`；标准非零 decay |
| Cautious | `p=-1e-30,m1=+1e-30`；两符号均为 NaN 比较为 false 的情况；signed zero |
| AMSGrad | old NaN/new finite、old finite/new NaN、相等、±0；首步 NaN；非 AMSGrad state→AMSGrad 时以新 m2 初始化；含 max state→非 AMSGrad 时保留旧 max 但不用于 delta |
| bias | 两个参数因缺梯度产生不同 time，不能共享 bias 分母 |
| 特殊值 | NaN、±Inf、±0、subnormal；检查分类和选择规则 |
| dtype | F32 按容差；F16/BF16/F64/Flex32/量化验证 Reference 回退且专用计数不增加；在无 `SHADER_F64` 的设备上编译并运行 F32，确认 launch 标量不引入 F64 |
| half evaluation | 首版在 `Chain`、`Accumulators` 下均以 F16 `grad=300, beta_2=0.999` 验证 Reference 的 `m2=Inf` 和输出，且专用计数不增加；未来开放半精度时还要覆盖 `PerOperation` 与所有中间舍入边界 |
| shape/layout | rank 0/1/2/4、含 1 维、transpose/narrow、空张量、tiled 回退 |
| alias | param==grad、共享 state、m1==m2、max 与 m2 同 allocation；P1 只读安全，P1b 不错误原地复用 |
| clipping | Value、逐参数 Norm、F16 norm 防溢出；证明没有折进 lr |

参考结果在同一 device、同一 dtype 下由 `reference_step` 生成，避免把不同后端的 NaN/sqrt 差异误判为 fused 错误。

### 11.2 模块与 decorator

- 混合 rank 模型，参数 shape 和 rank 均能恢复；
- 两个参数组使用不同 AdamW 配置和不同 lr，验证“最后匹配组”不变；
- 某些参数无梯度，其参数和 time 不变，其他参数正常更新；
- `step_multi`；
- Autodiff 的 require-grad、distributed、checkpointing strategy 在 step 后保持；
- reparameterized 参数经过 consume/restore 后行为不变；
- Fusion、Remote、Capture、Group 各自确认 reference 回退，不触发“Backend not supported”；
- raw Cube graph capture/replay 若当前 runtime 支持，比较捕获与 eager 输出；
- checkpoint 按第 8.3 节双向恢复。

### 11.3 Foreach 特有

- 同组不同 shape/rank；
- rank 0 和 rank 8 边界可正常恢复；rank 9 首步必须在 CollectMapper 的原 `step_dyn` 边界失败，专用 launch 计数为 0 且不生成 pending/state；对手工构造的 rank 9 `DynState`，后续 step、`to_device_dyn`、checkpoint flatten/unflatten 也保持原动态分派拒绝行为，不得出现首步成功、后续才 panic；
- 同 device/service 不同有效 stream 不得合并；自定义 mapper 在 A 的 `StreamId::executes` 中收集、退出后外层切到 B，观测必须证明输出分配和实际 launch 仍在 A，并由 A 上后续消费者读到完整更新；显式 pinned client 同样覆盖，无法查询有效 stream 的 revision 验证 Foreach→Reference；
- 同 shape 不同 lr/time 不得错误合并；
- 超参数相同但 state 一项已初始化、一项未初始化；
- 空 batch、单 item、长短 tensor、一个 tensor 跨多个 chunk、尾 chunk 非满；至少一个用例取 `numel = 2 * chunk_len + 3`，逐元素验证两个完整 chunk 和尾 chunk 都恰好更新一次，不能只检查 launch 成功；
- 超过 tensor/chunk 上限时拆分，所有元素恰好更新一次；
- chunk descriptor 改变时 JIT key 不碰撞；
- P2 out-of-place 下不同 tensor 共享 allocation 不覆盖输入；
- 证明发生两次 module map 时 mapper、path 和 state 各恢复一次。

不为测试扩大生产 API；测试通过 crate 内私有 helper、现有 public optimizer API 和现有 backend test aliases 到达行为。

## 12. 验证命令与报告要求

实施时按实际 backend alias 调整命令，并明确硬件不可用的项目：

```sh
# 默认参考路径与仓库回归
cargo test -p burn-optim --features burn-core/flex
cargo run-checks

# Flex+Fusion、没有 Cube runtime：weak 转发不得拉入 burn-cubecl
cargo test -p burn-optim --no-default-features --features std,fusion,burn-core/flex

# CubeCL CPU 的 reference/fused/foreach 与 compile-only（无观测接口）
# 仓库级检查使用上面的 cargo run-checks（当前默认 Flex），不使用不存在的 --backend cpu alias
cargo test -p burn-optim --no-default-features --features std,cpu

# raw/Fusion 路径证明与 launch/chunk 计数（feature 转发到 burn-cubecl）
cargo test -p burn-optim --no-default-features --features std,cpu,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,cpu,fusion,adamw-bench-observability

# 以下仅执行实际可用的 runtime，其余明确记为 skipped。
# 每个 runtime 都直接运行 burn-optim；run-checks 不能替代这些数值与 launch/chunk 路径断言。
# CUDA raw 与 Fusion 回退
cargo test -p burn-optim --no-default-features --features std,cuda,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,cuda,fusion,adamw-bench-observability
cargo run-checks --backend cuda

# ROCm raw 与 Fusion 回退（burn-optim::rocm 转发到 burn-cubecl::hip）
cargo test -p burn-optim --no-default-features --features std,rocm,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,rocm,fusion,adamw-bench-observability
cargo run-checks --backend rocm

# Metal raw 与 Fusion 回退
cargo test -p burn-optim --no-default-features --features std,metal,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,metal,fusion,adamw-bench-observability
cargo run-checks --backend metal

# WGPU raw 与 Fusion 回退
cargo test -p burn-optim --no-default-features --features std,wgpu,adamw-bench-observability
cargo test -p burn-optim --no-default-features --features std,wgpu,fusion,adamw-bench-observability
cargo run-checks --backend wgpu

# Fusion 普通构建（无观测接口）
cargo test -p burn-optim --no-default-features --features std,cpu,fusion

# 先验证独立 benchmark 确实进入专用路径，再用无观测开销的构建计时
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu,adamw-bench-observability -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu
```

直接 `burn-optim` 测试必须选择与 feature 对应的 device，并在 raw 配置断言专用 launch/chunk 增加、在 Fusion 配置断言不增加；不能用另一后端的输出代替。`cargo run-checks --backend ...` 的 backend tests 当前覆盖 `burn-backend-tests`、`burn-linalg` 和 `burn-signal`，并不运行这些新增的 optimizer 测试，因此只是附加回归，不是 GPU 快路径证明。

文档和 feature 变更后检查 `Cargo.lock`，不得出现无关依赖变化。格式化、clippy、feature/no-std 检查由 `cargo run-checks` 覆盖。若只实现 P1，不得把 P2 命令或未运行平台写成已验证。

## 13. 性能验收标准

不在设计阶段预设“必须快 60 倍”。每个 PR 使用同机同进程 A/B，并报告分布：

### P1

- 微小混合 rank 模型的稳态 optimizer-step 总时间应有统计上明确的改善；
- 大参数场景不得因标量首版出现不可接受的吞吐回退；若有，先完成向量化或限制快路径到 launch-bound 规模；
- 稳态不应再有参考路径的 13 个逻辑中间 tensor；输出分配数量与峰值内存单独报告；
- 首次 JIT 成本单独列出，不用稳态数字掩盖；
- Fusion 默认用户仍走 reference，不得把 raw eager 的改善宣传为全模式改善。

### P1b

- 共享输入绝不被修改；
- 稳态参数/state 可复用时分配和 drop 时间下降；
- 若 COW 判断成本超过收益，保留 P1 out-of-place，不强行合并。

### P2

- 与 P1 单参数 fused 比较，而不只与 Reference 比较；
- launch-bound 小参数模型在计入两次 module map 后有净收益；
- 大参数吞吐不得因 unroll/分支/chunk 退化；
- 超过上限的模型没有失控 JIT；
- 若收益只在某一 runtime 成立，快路径选择按实测能力限制，不宣称全平台收益。

性能 PR 的 changelog 只写实际测得的改进，附硬件和 workload。

## 14. PR 拆分、风险与退出条件

### PR 1：Reference 整理与 benchmark

只移动原公式、加入 execution 默认 Reference 和测试/benchmark。机械移动与行为变化分开提交，确认默认输出不变。

### PR 2：P1 out-of-place Fused

加入单参数内核、桥接、feature、数值/布局/decorator/checkpoint 测试。COW 留到后续，降低审查面。

### PR 3：P1b COW（有实测需要才做）

复用现有 `can_mut`/non-overlapping 规则和固定 input-position alias，加入所有权回归。

### PR 4：P2 Foreach prototype

先以 2～3 个 tensor 验证 `Sequence<LinearView>` 和 chunk unroll 能编译。若该 API 组合不成立，停止并重新提交只针对多参数参数绑定的小型设计，不退回物理拼接，也不留下不可编译的“生产级”代码。

### PR 5：P2 模块接入与分块调优

实现两次 map、完整分组、上限拆分、checkpoint 和性能验证。P2 任何失败都不回滚 P1。

### 退出条件

出现以下情况时停止对应阶段，而不是放宽正确性：

- Cautious/AMSGrad 的 NaN 或 signed-zero 语义无法保持一致；
- 必须修改共享输入才能取得收益；
- 无法在无 D2H 同步的情况下保留逐参数 lr/time；
- P2 的 JIT 或两次遍历在目标 workload 上持续抵消收益；
- 某 runtime 不支持所需参数形式；该平台保留 Reference，不用未验证代码兜底。

## 15. 参考资料

### Burn 当前源码

- `crates/burn-optim/src/optim/adamw.rs`：当前公式、配置和 state。
- `crates/burn-optim/src/optim/module/module_optimizer.rs`：参数组、clipping、lr、state 和恢复流程。
- `crates/burn-optim/src/grad_clipping/base.rs`：逐参数 Norm/Value 语义。
- `crates/burn-tensor/src/tensor/api/extension.rs`：`try_into_primitive` / `from_primitive`。
- `crates/burn-cubecl/src/kernel/unary_float.rs`、`binary.rs`：LinearView、COW、empty output、launch 方式。
- `crates/burn-cubecl-fusion/src/engine/codegen/ir.rs`：`Sequence<GlobalTensor>` 的实际用法。
- 锁定 CubeCL `sequence/base.rs`：Sequence 在编译期索引和展开的约束。

### 主流框架源码

- [PyTorch Adam 实现（AdamW 复用并启用 decoupled decay）](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py)
- [PyTorch AdamW 入口](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)
- [PyTorch AdamW 文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch CUDA MultiTensorApply](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh)
- [Optax AdamW 组合](https://github.com/google-deepmind/optax/blob/main/optax/_src/alias.py)
- [Optax scale_by_adam](https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py)
- [Optax add_decayed_weights](https://github.com/google-deepmind/optax/blob/main/optax/transforms/_adding.py)
- [Keras AdamW](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/adamw.py)
- [Keras Adam 更新公式](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/adam.py)
- [PyTorch gradient clipping 文档（说明 clipping 独立于 optimizer.step）](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

这些链接用于核对架构，不表示上述框架为 Burn 的数值或 API 兼容性提供保证。
