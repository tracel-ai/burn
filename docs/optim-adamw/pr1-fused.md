# PR 1：AdamW 单参数 out-of-place Fused

> 文档状态：已实现；实测数据见 [pr1-fused-report.md](pr1-fused-report.md)
>
> 性能类型：减少逻辑中间 tensor、算子分发和物理 launch
>
> 代码依赖：无，可基于最新 `main` 独立推进
>
> 直接性能对照：同一 revision、同一 device 上的 Reference

本文独立说明该 PR 的实现、正确性和性能门禁，不需要阅读其他 PR 文档。

## 1. 性能问题与独立价值

当前 raw Cube eager Reference 在单参数内依次更新一阶矩、二阶矩、可选 AMSGrad max、bias correction、weight decay 和参数。无 AMSGrad、无 cautious decay、state 已初始化时，可以数出 15 个高阶 tensor 算子。

这个计数只说明融合机会，不等于所有模式都有 15 次物理 launch、15 次新分配或固定 `15 read + 15 write`。CubeCL eager 可能原地复用，Fusion 和 graph replay 也会改变物理执行。因此本 PR 不预先宣称“减少 93% launch”或“显存往返降低 73.3%”，而是用分派测试、profiler 和端到端 benchmark 给出结果。

本 PR 交付一次单参数 F32 CubeCL 内核，完成完整 AdamW 更新，使用 out-of-place 输出；不支持时完整回退 Reference。

独立收益假设：

- 减少单参数更新的逻辑中间 tensor；
- 减少 host 算子分发和潜在物理 launch；
- 在微小及混合 rank 模型上降低 optimizer-step 总时间。

本 PR 不包含 COW、多参数 Foreach、向量化或 Fusion CustomOpIr。它们是否值得做由其他独立 PR 判断。

## 2. 基线与兼容边界

当前实现位于 `crates/burn-optim/src/optim/adamw.rs`：

| 项目 | 当前事实 | 本 PR 约束 |
|---|---|---|
| `beta_1` / `beta_2` | `0.9` / `0.999` | 不变 |
| `epsilon` | `1e-5` | 不改成 PyTorch 的 `1e-8` |
| `weight_decay` | `1e-4` | 不改成 PyTorch 的 `1e-2` |
| `amsgrad` | 默认 `false` | Fused 必须支持 |
| `cautious_weight_decay` | 默认 `false` | Fused 支持；所有乘法使用显式舍入，避免 FMA contraction 翻转 mask |
| `grad_clipping` | 已存在 | 保留逐参数语义 |
| `LearningRate` | `f64` | Host 精度不变 |
| state | 每参数 `AdaptiveMomentumState<D>` | rank、dtype、time、名称不变 |

`ModuleOptimizerMapper::map_float` 的参数组选择、逐参数 clipping、逐参数 lr、无梯度跳过、ParameterContext 和 ParamMapper 恢复全部不变。本 PR 只替换 inner backend 上的数值更新方式。

不扩大核心 `Backend` trait，不修改 checkpoint 格式，不实现全局 clipping、FP32 master weights 或新 AdamW 算法。默认 Reference 保持原行为，只有显式选择 Fused 才尝试专用路径。

## 3. 数值兼容契约

### 3.1 Host 标量与矩更新

令 `t` 为当前参数自己的 state time，Host 按现有 Rust 代码计算：

```text
time_i32   = state.time as i32
factor_1   = 1.0f32 - beta_1
factor_2   = 1.0f32 - beta_2
bias_1     = 1.0f32 - beta_1.powi(time_i32)
bias_2     = 1.0f32 - beta_2.powi(time_i32)
decay_rate = lr * (weight_decay as f64)
```

已有 state 时，按 tensor dtype `F` 执行：

```text
m1 = old_m1 * F(beta_1) + g * F(factor_1)
m2 = old_m2 * F(beta_2) + (g * g) * F(factor_2)
```

首步不读取未初始化 old state：

```text
m1 = g * F(factor_1)
m2 = (g * g) * F(factor_2)
max_m2 = amsgrad ? m2 : absent
```

已有 state 时，max state 与 `has_state` 独立：

```text
if amsgrad:
    previous_max = has_old_max ? old_max : m2
    new_max = (previous_max < m2) ? m2 : previous_max
    v_for_update = new_max
else:
    new_max = has_old_max ? old_max : absent
    v_for_update = m2

m1_corrected = m1 / F(bias_1)
v_corrected  = v_for_update / F(bias_2)
delta = m1_corrected / (sqrt(v_corrected) + F(epsilon))
```

必须同时记录 `has_state` 和 `has_old_max`。从非 AMSGrad checkpoint 切换到 AMSGrad 时，已有 state 但无 old max，本次新 m2 同时充当 previous max 和结果；反向切换时保留已有 `Some(old_max)` 及其 buffer/值，但本次不更新、不用于 delta。

禁止用 `1/bias` 后乘法替代除法，也禁止改成 Keras 的 `sqrt(bias_2)/bias_1` 形式；代数相近不代表舍入顺序相同。

### 3.2 Decay 与 AMSGrad 特殊值

Host 根据 `decay_rate == 0.0` 选择：

```text
NoDecay:
    decayed = p

Standard:
    decayed = p * F(1.0f64 - decay_rate)

Cautious:
    apply_decay = ((p >= 0) == (m1 >= 0))
    raw_decay = p * F(decay_rate)
    decayed = p - (apply_decay ? raw_decay : F(0))

p_new = decayed - delta * F(lr)
```

- NoDecay 跳过 decay 乘法，避免引入 Reference 不存在的 `0*Inf`。
- Cautious 分别比较符号，不能改成 `(p*m1)>=0`；乘积下溢为 `-0.0` 会改变 mask。
- Cautious 使用更新后的 m1；weight decay 不进入梯度或动量。Reference 的每个 tensor 乘法都会单独物化和舍入；内核用 `mul(a,b)=fma(a,b,negative_zero)` 实现同样的单独舍入，其中 `negative_zero` 是运行时标量而非编译期字面量（WGSL 会把字面量折叠回乘法并与后续减法收缩）。这样 `m1` 和 decay 乘积与 Reference 的舍入边界一致，cautious mask 可以直接在内核中执行。
- AMSGrad 复现 `old_max.max_pair(m2)`：mask 为 `old_max < m2`，true 选新值，false 保留 old。
- old NaN 被保留；new NaN 不覆盖有限 old；相等或 `±0` 比较为 false 时保留 old 位模式。
- 首步或已有 state 但无 old max 时直接以新 m2 初始化，因此该情况下 NaN 保留。
- 不假定硬件 `max`、`fmax` 或 `F::max` 具有相同 NaN/signed-zero 规则。

### 3.3 Clipping 不折进学习率

clipping 继续在 Fused 前由原 module mapper 执行。`Value` 是逐元素截断；`Norm` 对每个参数分别求 L2 norm，F16/BF16 先转 F32，系数为 `threshold/(norm+min_positive)` 后 `clamp_max(1.0)`，再乘梯度并 cast 回原 dtype。

若 `g'=alpha*g`，必须先用 g' 更新 m1/m2，再计算 delta；weight decay 仍依赖原参数和原 lr。禁止把 alpha 折进 lr。clipping on/off 分开 benchmark。

### 3.4 精度、dtype 与布局

- Host 标量保持 f32/f64 精度，只在进入内核时转换为 F。
- 初版使用标量 `Const<1>`，不做向量重排、不归约、不做代数重组。唯一的 `fma` 是 `mul` 的舍入屏障：`fma(a,b,negative_zero)` 的值等于单独舍入的乘积，`negative_zero` 必须来自运行时标量。
- 编译器可能 FMA contraction，不同目标的除法、sqrt、subnormal 和 NaN 指令也可能不同，因此承诺数值容差兼容，不承诺 bitwise 相同。
- F32 初始建议 `atol=4ε`、`rtol=8ε`，由 `dtype.finfo()` 推导并审查误差来源；不得为通过测试直接放大容差。
- NaN/Inf、signed zero 和 AMSGrad 选择规则另做分类或逐位断言。

首版专用路径只开放 **F32 非量化输入**，要求 param/grad/m1/m2/max 的 shape、device、dtype 一致。F16、BF16、F64、Flex32、量化、不同 state dtype、空张量和 storage-tiled 输入回退 Reference。

半精度不能只靠容差开放。CubeCL `F16Evaluation::Chain` 会把一串 F16 算术保持在 F32，`Accumulators` 还会扩大保持范围，而 Reference 的每个 tensor op 都有独立 dtype 物化边界。例如 F16 首步 `grad=300,beta_2=0.999`：Reference 先把 `300*300=90000` 物化为 F16 Inf，再得到 m2 Inf；fused 若整段保持 F32，会得到有限 `m2≈90`，同时改变 state 和参数。这不是 FMA 或普通容差可以覆盖的差异。

非连续但非重叠输入通过 LinearView 读取，不强制 reshape 或 into_contiguous。输出使用连续、同 shape 的新 tensor；rank 1/2/3/4、transpose 和 narrow 都加入测试。rank 0 在 Fused 与 Reference 都不可用（Reference 的 elementwise launch 用 `num_dims() - 1`，会 panic），Fused 在准入时直接拒绝。空张量在 Host 回退 Reference，保持 time 推进和 state 初始化语义。

## 4. 并入本 PR 的准备工作

以下内容是性能实现和证据的一部分，不另开 benchmark-only 或 refactoring-only PR。

### 4.1 固定 Reference

把当前 `AdamW::step` 和 `AdaptiveMomentumW::transform` 原函数体整理为私有 `reference_step`，先运行现有测试，再接入分派。Reference 和 candidate 使用同一数学锚点，默认配置不走新代码。机械移动与内核行为变化分 commit，便于审查，但同属这个性能 PR。

### 4.2 执行配置

使用一个枚举避免 `fused=false, foreach=true` 的无定义组合。本 PR 只公开已经实现的 Reference 和 Fused，不提前暴露 Foreach：

```rust
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum AdamWExecutionConfig {
    /// 当前逐步 tensor 实现；默认值。
    Reference,
    /// 每个参数一次融合内核；不支持时回退 Reference。
    Fused,
}
```

枚举从引入起使用 `#[non_exhaustive]`，为以后新增执行策略保留源兼容空间。只有真实多参数路径随其实现 PR 一起交付时，才增加 Foreach variant；本 PR 不留未实现的 public API。

`AdamWConfig` 新增 `execution: Option<AdamWExecutionConfig>`。旧 JSON/binary 缺少字段时，Serde 对 Option 得到 None，`build()` 将 None 解析为 Reference；`with_execution(...)` 写入 Some。`AdamW` 保存解析后的非 Option execution。直接调用 `Optimizer::step` 并选择 Fused 时使用单参数融合路径；但若 tensor 仍在 autodiff context 中，primitive 转换失败并回退 Reference（模块路径由 `tensor.inner()` 去除该 context）。

增加上一版本配置的 JSON 与 `load_binary` 回归测试，以及显式 Fused 的 round-trip。不能把只影响 `new()` 的 `#[config(default)]` 当成反序列化迁移机制。新增 public API 更新 rustdoc；本分支不修改 changelog。

### 4.3 Benchmark

新增 `burn-optim` 自有 optimizer-step benchmark，并证明 benchmark 实际进入专用路径。测试和 benchmark 通过 CubeCL 现有的 `LaunchObserver` 观察调用线程实际提交的 AdamW kernel；观测辅助代码仅在测试和 benchmark 中编译，不增加生产计数器或 feature：

- `crates/burn-optim/benches/adamw.rs` 用 Divan 比较 Reference 与 Fused：微小模型、混合 rank、1/4/16 个大参数以及长短混合；计时区分 enqueue 与 enqueue + `Device::sync`。
- `main` 断言 `!burn_cubecl::FUSED`。Fusion 构建直接失败，避免把回退后的 Reference 数字标成 Fused 结果。
- `--verify-path` 在 `divan::main()` 前由 target 自己解析，对每个 workload 先运行 Reference 再运行 Fused 并比较数值；`assert_fused_runs` 在 Initial 与 State 两次生产 `optimizer.step` 前后读取 launch 计数，差值必须等于参数个数，回退会让断言失败。case 清单非空且全部完成才退出成功。
- 路径分派本身由测试覆盖：同一组数值测试在 raw 与 Fusion 两种构建下运行；raw 构建的
  `fused_boundary_accepts_initial_and_state` 直接调用生产边界 `try_adamw`，断言 Initial 和 State
  输入都返回 `Ok`（不是回退）。该测试需要 `--features std,cpu` 的 raw 构建。

## 5. 文件与 feature 组织

建议变更：

```text
crates/burn-cubecl/src/kernel/optim/mod.rs       # 新模块
crates/burn-cubecl/src/kernel/optim/adamw.rs     # launch、kernel、输出规划
crates/burn-optim/src/optim/adamw.rs             # reference + execution 分派
crates/burn-optim/src/optim/adamw/cube.rs        # 高层 Tensor 桥接
crates/burn-optim/src/optim/mod.rs               # execution 配置重导出
crates/burn-optim/Cargo.toml                     # 可选 burn-cubecl、runtime features、bench
crates/burn-optim/build.rs                       # runtime features → `cubecl_backend` cfg
crates/burn-optim/benches/adamw.rs               # Divan 与 path-verification
crates/burn/Cargo.toml                           # weak 转发 burn-optim runtime features
```

`burn-optim` 把 burn-cubecl 设为可选依赖并 `default-features=false`，避免默认 Fusion 意外关闭 raw 快路径；现有 `std` feature 追加 `burn-cubecl?/std`。

`burn-optim` 增加 runtime feature：`cpu/cuda/rocm/wgpu/wgpu-msl/vulkan/webgpu/metal`，每个同时启用 `std`、`dep:burn-cubecl`、`burn-core/<runtime>` 和 `burn-cubecl/<runtime>`（`rocm` 映射到 burn-cubecl 的 `hip`）。burn-cubecl 本身依赖 std，因此 runtime feature 隐含 std。`build.rs` 在没有任一 runtime feature 时不设置 `cubecl_backend` cfg，桥接模块因此不编译，整个 crate 仍可构建并走 Reference；不再提供可单独启用的内部 `cubecl-backend` feature。

`burn` 的每个 CubeCL backend feature 增加 weak 转发 `burn-optim?/<runtime>`，因此 facade 用户启用 backend 时同时编入桥接代码；`fusion` feature 不转发。`burn` 的默认 features 会启用 kernel fusion（`burn-core/default`），该组合下 Fused 仍回退 Reference，所以实际运行内核需要 `default-features = false` 的 raw 构建（例如 `--no-default-features --features std,optim,cpu`）。

benchmark target 在 dev-dependencies 中加入仓库现有版本的 Divan，并显式配置：

```toml
[[bench]]
name = "adamw"
path = "benches/adamw.rs"
harness = false
```

计时入口调用 `divan::main()`。raw 测试/benchmark 必须关闭整条默认 feature 链：命令使用 `--no-default-features --features std,cpu`，同时 `burn-nn` dev-dependency 改为 `default-features=false, features=["std"]`，否则测试依赖会重新打开 `burn-core/default -> burn-cpu?/default -> burn-cubecl/default -> fusion`。Fusion 回退另用 `std,cpu,burn-core/fusion`，不与 raw 配置共用结论。Flex benchmark 使用 `std,burn-core/flex,burn-core/default` 保留 Flex 的 simd/rayon 默认项。

不修改 `burn-backend::Backend`，因此 Flex、Remote 和自定义 backend 不需要新增 AdamW 方法。

## 6. 分派、转换与输出

```rust
match self.execution {
    Reference => self.reference_step(lr, tensor, grad, state),
    Fused => {
        match cube::try_step(lr, tensor, grad, state, &self.momentum, self.weight_decay,
                             self.cautious_weight_decay) {
            Ok(updated) => updated,
            Err(original_inputs) => self.reference_step_with(original_inputs),
        }
    }
}
```

`try_step` 先检查 execution、dtype、shape、device、空张量和 layout，再尝试 primitive 转换。所有回退必须发生在 launch 前；内核部分执行或 launch 失败后不能换算法再执行一次。

`try_into_primitive` 会消费 tensor 且错误不携带原 tensor，因此桥接采用两阶段转换：

1. 先做 device/kind/dtype 检查；
2. 对仍保留的原始 Tensor 做句柄级 clone，用 clone 尝试转换；
3. param/grad/state 全部成功后才 drop 原始句柄并 launch；
4. 任一失败则 drop 已转换 clone，用完整原输入走 Reference；
5. burn-cubecl `try_adamw` 在 Fusion 形态返回原 primitive，桥接恢复 Tensor 后走 Reference。

clone 只增加临时句柄所有权，不复制 tensor 数据。

burn-cubecl 提供小型 `try_adamw` 边界：`not(feature="fusion")` 接收 raw CubeTensor 并执行专用内核；`feature="fusion"` 原样返回所有 primitive。burn-optim 不猜测 Cube feature 形态，也不伪造 Fusion IR。未来进入 Fusion 需要另一个 CustomOpIr 设计，声明全部读写、alias、capture 和 Remote handler。

执行模式：

| 模式 | 本 PR 行为 |
|---|---|
| Reference | 原实现，不检查 Cube 快路径 |
| raw Cube eager | 合格输入进入专用内核 |
| Autodiff | 模块路径沿用 `tensor.inner()` 和 ParameterContext restore，更新不建反向节点；直接 `Optimizer::step` 收到 autodiff context 的 tensor 时回退 Reference |
| Fusion | 返回原输入并走 Reference |
| Remote / Capture / Group | primitive 不匹配时走 Reference |
| 量化 / 不支持 dtype/layout | 回退 Reference |
| `step_multi` | 沿用 GradAdaptor 梯度合并/device 结果后更新 |

逻辑输出为：

```rust
struct AdamWOutput {
    param: CubeTensor,
    moment_1: CubeTensor,
    moment_2: CubeTensor,
    max_moment_2: Option<CubeTensor>,
}
```

所有输出保留 shape、strides 语义、dtype 和 device。桥接在原 D 下构造 Tensor 和 AdamWState，Host 只推进原 state.time，因此 RecordState、Burnpack 名称和 `__rank` 不变。

本 PR 为 param/m1/m2 以及需要更新的 AMSGrad max 分配独立输出。param、grad 和 state 即使共享存储，内核也只读输入，不会中途覆盖后续读取。非 AMSGrad 的旧 max 不更新、不分配新副本，按 Reference 原样保留。

## 7. 内核与 launch

实现采用与 `unary_float.rs`、`binary.rs` 相同的机制：LinearView/LinearViewMut、`output.is_in_bounds(ABSOLUTE_POS)`、越界 `terminate!()`、`calculate_cube_count_elemwise`、`address_type="dynamic"`、`dtype_to_storage_type` 和 `Const<1>`。不使用当前 API 不支持的 `Tensor::is_in_bounds/read/write` 组合。

Host 与 launch 标量分开：

```rust
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
    negative_zero: InputScalar,
}

enum DecayMode { None, Standard, Cautious }
```

旧 state、旧 max 和 max 输出用 `ComptimeOption<LinearView>` / `ComptimeOption<LinearViewMut>` 表示。内核用 `#[comptime] if let ComptimeOption::Some(...)` 在展开期选择分支，因此每个组合仍是独立 kernel variant，缺失的 view 不会绑定，不产生 dummy view。这样不需要为 init/update/AMSGrad 迁移分别写内核函数。只有 `DecayMode` 是 comptime；数值标量都是 runtime `InputScalar`，lr schedule 不触发每步 JIT。

bias 是分母，不是倒数。F32 kernel 的所有 InputScalar 按 F32 创建，尤其 Host 为 f64 的 lr/decay 标量不得引入 F64 类型或 `SHADER_F64` 要求。

无旧 state 或无 AMSGrad 时不读取未初始化内存：没有 `Some` view 的分支不会展开。`amsgrad && !has_old_max` 时 `old_max` 为 `None`，内核把新 m2 同时作为 v；`!amsgrad && has_old_max` 时 Host 不传 max view，内核不更新 max，Host 原样保留旧 max。

内核核心伪代码（`Some`/`has_state` 都是 comptime；`mul(a,b,negative_zero)` 是运行时加数的舍入屏障，让每个乘积单独舍入）：

```rust
let m1 = if old_moment_1.is_some() {
    mul(old_m1.read(i), beta_1, negative_zero) + mul(g, factor_1, negative_zero)
} else {
    mul(g, factor_1, negative_zero)
};
let m2 = if old_moment_2.is_some() {
    mul(old_m2.read(i), beta_2, negative_zero) + mul(g * g, factor_2, negative_zero)
} else {
    mul(g * g, factor_2, negative_zero)
};
let v = if let ComptimeOption::Some(old_max) = old_max_moment_2 {
    select(old_max.read(i) < m2, m2, old_max.read(i))
} else {
    m2
};
if let ComptimeOption::Some(mut out_max) = out_max_moment_2 {
    out_max.write(i, v);
}
let delta = (m1 / bias_1) / ((v / bias_2).sqrt() + epsilon);
let decayed = match comptime![decay] {
    DecayMode::None => p,
    DecayMode::Standard => mul(p, one_minus_decay_rate, negative_zero),
    DecayMode::Cautious => {
        let same_sign = (p >= zero) == (m1 >= zero);
        p - select(same_sign, mul(p, decay_rate, negative_zero), zero)
    }
};
out_param.write(i, decayed - mul(delta, lr, negative_zero));
```

`supported` 还检查 `num_bindings`（输入、输出、可选 state/max 和 metadata buffer）不超过
`client.properties().hardware.max_bindings`，避免在绑定数不足的 runtime 上提交必然失败的
launch。

实施时核对当前 CubeCL 的 LaunchArg、`InputScalar::get`、select 和 comptime 宏语法；伪代码不能直接作为已编译代码引用。

## 8. 正确性测试

先证明目标路径实际执行，再比较数值。Reference 结果在同一 device、同一 dtype 下生成，避免跨后端 NaN/sqrt 差异。

| 类别 | 已覆盖的用例 |
|---|---|
| 基础 | 首步、5 步、50 步（均走内核）；正弦分布的正负梯度；cautious/AMSGrad 四种组合；param/m1/m2/max 全部比较 |
| decay | `weight_decay=0` 且 `p=Inf`（NoDecay 不做乘法）；标准非零 decay |
| Cautious | 内核执行且与 Reference 一致（含特殊值、AMSGrad 组合，逐位断言 ±0/Inf、分类 NaN） |
| AMSGrad | old/new NaN、±0；无 max state→AMSGrad；含 max state→非 AMSGrad |
| bias | 不同 state time（1、3、5）使用各自的 bias 分母；同一 step 内多参数共享同一公式 |
| 特殊值 | NaN 分类、±Inf、±0 位模式、subnormal |
| dtype | F32 容差；F16/BF16/F64 按 `supports_dtype` 实际执行回退测试；F16 `grad=300` 的 Reference 物化边界 |
| shape/layout | rank 1/2/3/4、transpose、narrow、空张量回退；rank 0 转换后由 `try_adamw` 拒绝 |
| alias | param==grad、m1==m2、max 与 m2 同 allocation，验证 out-of-place 只读安全 |
| clipping | Value、逐参数 Norm；weight 和 bias 都参与比较 |
| module | 混合 rank 的 Linear（2D weight + 1D bias）、Autodiff device、parameter group 之外的原路径 |
| decorator | Autodiff（模块路径）；Fusion 构建回退且数值一致 |
| checkpoint | Reference 保存→Fused 恢复、Fused 保存→Reference 恢复；AMSGrad 开关 |

未单独覆盖（由 dtype/layout 准入统一回退，后续可补）：Flex32、量化、storage-tiled、Remote/Capture/Group、`step_multi`、reparameterized 参数。rank 0 在 Fused 准入时被拒绝，Reference 本身在该 rank 上会 panic。

checkpoint 验收检查每个 ParamId 的 `__rank`、time、tensor 名称、shape、dtype，加载后 group/path 和逐参数 lr 不变。不同执行方式的浮点 state 按容差比较，不要求 bitwise 相同。

## 9. Benchmark 与性能验收

### 9.1 测量对象

至少报告：

1. Host enqueue，计时区内不 sync；
2. enqueue + `Device::sync` 总时间；
3. 首次 JIT/编译，与稳态分开；
4. step 作用域结束后的 drop/sync；
5. 参数元素数、总字节数、参数个数、launch/allocation 和峰值内存（可获得时）。

统一 workload：

| 模型形态 | 参数张量 | 诊断对象 |
|---|---|---|
| 微小模型 | 30 个 `[64,64]`，F32 | launch/host 开销 |
| 混合 rank | bias `[64]`、linear `[64,64]`、conv `[16,8,3,3]` | rank 恢复与混合参数 |
| 大参数 | 1、4、16 个 `[1024,1024]` | 标量内核吞吐回退 |
| 长短混合 | 多个小 bias 与 1～4 个大矩阵 | 不同参数规模 |

F16/BF16 只验证 Reference 回退，不纳入首版加速验收。记录 hardware、OS、Rust/Cargo feature、fusion on/off、编译模式、样本数、warmup、median/p95 和原始结果。Batch size 不进入纯 optimizer-step benchmark。

A/B 使用同机、同 runtime、同 features，只改变执行方式。报告稳态总时间并拆分 enqueue/device/drop/JIT，防止把成本移动到计时区外。

### 9.2 合并门禁

必须同时满足：

- 微小混合 rank 模型的稳态 optimizer-step 总时间有统计上明确的改善；
- 与 Reference 对比列出 launch、逻辑中间 tensor、allocation 和峰值内存的实测变化；
- 大参数场景没有不可接受的吞吐回退；若标量内核只在 launch-bound 规模有收益，本 PR 可用数据支持的阈值保守限制快路径，不能把明显回退留给后续 PR；
- 首次 JIT 单独列出；
- Fusion 默认用户仍走 Reference，不把 raw eager 改善宣传为全模式改善。

如果目标 workload 没有净收益，或必须改变数值语义才能变快，则不合并本 PR。

## 10. 验证命令

按实际可用 runtime 执行；不可用硬件明确标为 skipped。

```sh
# 默认 Reference 与仓库回归
cargo test -p burn-optim --features burn-core/flex
cargo run-checks

# Flex+Fusion、没有 Cube runtime：不得拉入 burn-cubecl
cargo test -p burn-optim --no-default-features --features std,burn-core/flex,burn-core/fusion

# CubeCL CPU raw 与 Fusion
cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu,burn-core/fusion

# 先验证 benchmark 路径，再采集性能
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu

# CUDA raw/Fusion；其他 runtime 按硬件替换并如实报告
cargo test -p burn-optim --no-default-features --features std,cuda
cargo test -p burn-optim --no-default-features --features std,cuda,burn-core/fusion
cargo run-checks --backend cuda
```

ROCm 使用 `rocm`，Metal/WGPU 使用对应 feature；`rocm` 在 burn-optim 内映射到 burn-cubecl 的 `hip`。`cargo run-checks --backend ...` 不运行新增 optimizer 测试，只是附加回归，不是快路径证明；快路径测试用上面的 `--features std,cpu` 命令单独运行。仓库级检查当前默认 Flex，不使用不存在的 `--backend cpu` alias。

检查后审阅生成 diff，确保 `Cargo.lock` 没有无关依赖变化。未运行平台不得写成已验证。

## 11. PR 交付清单与停止条件

PR 描述至少包含：

- 单参数 Reference 的性能问题和直接 A/B；
- 新执行配置及旧配置兼容；
- Fused 范围、回退矩阵和明确不做的 COW/Foreach/向量化；
- 数值、布局、decorator、checkpoint 和分派/回退测试结果；
- 稳态/JIT/enqueue/device/drop 数据及原始结果链接；
- 实测平台、跳过平台和 AI Usage。

出现以下情况停止本 PR：AMSGrad 特殊值无法保持一致、必须引入 D2H sync、必须修改共享输入才能取得收益，或目标 workload 没有端到端净收益。

## 12. 实现参考

- `crates/burn-optim/src/optim/adamw.rs`：当前公式、配置和 state。
- `crates/burn-optim/src/optim/module/module_optimizer.rs`：参数组、clipping、lr、state 和恢复流程。
- `crates/burn-tensor/src/tensor/api/extension.rs`：primitive 转换边界。
- `crates/burn-cubecl/src/kernel/unary_float.rs`、`binary.rs`：LinearView 和 launch 方式。
- [PyTorch AdamW](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)：single-tensor、fused、foreach 路径对照，不照搬其默认值或公式。
- [Keras Adam](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/adam.py)：用于确认不采用不同的 alpha 舍入顺序。
