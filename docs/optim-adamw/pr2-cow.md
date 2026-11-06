# PR 2：AdamW 参数与 state 的 COW 输出复用

> 文档状态：已实现并连续两轮通过 review；测量及共享CPU限制见 [pr2-cow-report.md](pr2-cow-report.md)。
>
> 性能类型：减少稳态分配、回收和峰值内存
>
> 代码依赖：PR1 的 out-of-place Fused 已完成连续两轮 review；不依赖 Foreach 或向量化。
>
> 直接性能对照：合入前的 out-of-place Fused，不是 Reference

本文独立说明该 PR 的实现、正确性和性能门禁，不需要阅读其他 PR 文档。

## 1. 性能问题与独立价值

单参数 Fused 使用 out-of-place 输出后，每个 step 仍需为 param、moment 1、moment 2 以及需要更新的 AMSGrad max 准备输出。若 profile 证明输出 allocation、drop 或内存压力占显著成本，可以在所有权和 alias 安全的前提下复用旧 buffer。

本 PR 可接受的独立收益是：

- 减少稳态输出 allocation 和回收；
- 在计入 COW/alias 判断后降低端到端 optimizer-step 总时间；或
- 显著降低峰值内存，同时总时间没有不可接受回退。

只减少分配次数，但总时间和峰值内存都没有实质改善，不足以合并。实施前先在 PR 描述中声明主要目标是时间还是内存，避免测完后挑选指标。如果 profile 没有显示分配或内存瓶颈，直接不推进本 PR。

## 2. 范围与非目标

本 PR 包含：

- 对 param/m1/m2/max 每个输出分别做复用资格判断；
- 合格时使用固定 input-position alias，不合格时保留 out-of-place；
- 共享输入、state alias、AMSGrad 和 checkpoint 回归；
- 与 out-of-place Fused 的直接 A/B benchmark。

本 PR 不包含：

- 多参数 Foreach；
- 向量化或 dtype 扩展；
- 第二套 Foreach 专用所有权 planner；
- 为了复用而修改共享输入；
- Reference 数学公式或配置语义变化。

如果 Foreach 已先合并，只能复用本 PR 已验证的同一 planner，并重新验证多参数场景，不能借本 PR 重写 Foreach。

## 3. 必须保持的数值与状态契约

COW 只改变输出存储位置，不改变 AdamW 计算。专用路径仍按 tensor dtype `F` 执行：

```text
m1 = old_m1 * F(beta_1) + g * F(1.0f32 - beta_1)
m2 = old_m2 * F(beta_2) + (g * g) * F(1.0f32 - beta_2)

if amsgrad:
    previous_max = has_old_max ? old_max : m2
    new_max = (previous_max < m2) ? m2 : previous_max
    v = new_max
else:
    v = m2

m1_corrected = m1 / F(1.0f32 - beta_1.powi(time as i32))
v_corrected  = v / F(1.0f32 - beta_2.powi(time as i32))
delta = m1_corrected / (sqrt(v_corrected) + F(epsilon))
```

首步不读取未初始化的 old state，而是 `m1 = g * factor_1`、`m2 = (g*g) * factor_2`；AMSGrad 首步以新 m2 初始化 max。

decay 保持三种模式：

```text
NoDecay:  decayed = p
Standard: decayed = p * F(1.0f64 - lr * weight_decay as f64)
Cautious: decayed = p - (((p >= 0) == (m1 >= 0)) ? p * F(decay_rate) : F(0))

p_new = decayed - delta * F(lr)
```

关键约束：

- Host 标量精度、bias 分母、运算顺序和当前 Reference 保持一致；不把除法改成 reciprocal 乘法。
- NoDecay 不引入 `0 * Inf`；Cautious 不改成 `(p*m1) >= 0`，并使用更新后的 m1。
- AMSGrad 的选择方向保持 `old_max < m2`：old NaN 被保留，new NaN 不覆盖有限 old，相等或 signed zero 比较为 false 时保留 old 位模式。
- 从非 AMSGrad state 切换到 AMSGrad 时，已有 state 但无 old max，本次新 m2 同时初始化 previous/result。
- 非 AMSGrad 遇到旧 `Some(max)` 时原样保留，不更新、不用于 delta、不擅自清空。
- param/m1/m2/max 的 shape、rank、dtype、device、state time、RecordState/Burnpack 名称和 `__rank` 不变。
- 首版仍只支持 F32 非量化输入；F16/BF16/F64/Flex32、量化、空张量、tiled 和其他不支持情况回退 Reference。
- 不承诺与 Reference bitwise 相同；F32 使用由 `dtype.finfo()` 推导且经审查的容差，特殊值另做分类或位模式断言。

## 4. 复用资格判断

每个输出独立判断，不能因为 param 可复用就推断 state 也可复用：

1. 输入 `can_mut()` 且 `is_nonoverlapping()`、`is_contiguous()`，不是 tiled；
2. 输入 allocation id 不与任何其他角色的 allocation id 相同；
3. 输入与输出的 shape、dtype、layout 相容；
4. 所有 primitive 转换成功后，原高层 tensor/state wrapper 已 drop；primitive 拒绝时从原 inputs 恢复 state/time/max；
5. alias 参数的 input position 与实际 launch 签名一致。

任一条件失败就分配新输出。`as_linear_view_alias` 只在全部条件满足时使用。

分配标识可比较 `handle.service` 与 `handle.memory.descriptor().id`。同一 allocation 的不同 slice 先保守视为 alias；只有未来能提供区间不相交证明时才放宽。

### 4.1 输入角色与写入安全

Fused 内核会读取 param、grad、old m1、old m2、可选 old max，再写 new param、new m1、new m2、可选 new max。若任何输入角色共享 allocation，在没有更精细区间和读写顺序证明时，不得把其中任一角色原地用作输出，否则可能覆盖后续读取。

临时 clone 会增加所有权计数。必须在 COW 判断之前 drop，否则独占输入会被误判为共享，造成无谓分配和性能回退。

### 4.2 AMSGrad 特例

- 首步以及已有 state 但 `max_moment_2=None` 时没有 old max，不得复用 dummy 输入，必须为新 max 分配输出。
- 非 AMSGrad 保留旧 max 的路径不参与 max COW 写入。
- 有 old max 且 AMSGrad 开启时，也只有在全部通用资格满足后才复用。

## 5. 正确性测试

通过 CubeCL 现有 LaunchObserver 证明生产路径，allocation ID 断言区分复用与分配；观测测试使用测试锁。每个输出单独判定，其他角色的共享不会阻止独立输出复用。部分 runtime 的 narrow 会生成连续副本，测试按真实 metadata 验证。

至少覆盖：

| 类别 | 用例与断言 |
|---|---|
| param 共享 | 外部仍持有 param 时不得修改共享值，选择新输出 |
| state 共享 | 外部仍持有 m1/m2/max 时不得覆盖旧 state |
| param==grad | 不原地覆盖后续读取，结果与 out-of-place 一致 |
| m1==m2 | 不错误原地复用，两个输出语义不串写 |
| max 与 m2 同 allocation | 不原地复用，AMSGrad 选择结果正确 |
| 可复用 | 独占且不相交时确实复用，param/m1/m2/max 数值一致 |
| layout | shape/dtype/layout 相容才复用；transpose/narrow/tiled 按规则处理或回退 |
| AMSGrad | 首步、无 old max、有 old max、非 AMSGrad 保留 old max |
| 特殊值 | old/new NaN、±0、Inf 的选择和分类不因复用改变 |
| 多步 | 50 steps 中与 out-of-place 比较参数和全部 state |
| checkpoint | Fused 保存→Reference 加载，以及反向恢复；time/rank/名称不变 |
| capture | runtime 支持时比较 graph capture/replay 与 eager |

Reference 和 candidate 在同一 device、同一 dtype 下比较，避免把跨后端 sqrt/NaN 差异误判为 COW 错误。

## 6. Benchmark 与性能验收

### 6.1 测量方法

复用现有 `burn-optim` AdamW optimizer-step benchmark，至少包含：

- 30 个 `[64,64]` F32 参数；
- bias `[64]`、linear `[64,64]`、conv `[16,8,3,3]` 混合 rank；
- 1、4、16 个 `[1024,1024]` 大参数；
- 强制共享输入而不可复用的对照 case；
- 独占输入而可复用的稳态 case。

分别报告：

1. Host enqueue，不在计时区 sync；
2. enqueue + `Device::sync` 总时间；
3. step 作用域结束后的 drop/sync；
4. allocation 次数、复用次数、回收次数和峰值内存；
5. 首次 JIT 与稳态分开；
6. median/p95、warmup、样本数和原始结果。

A/B 使用相同 hardware、runtime、Cargo features 和 workload，对照组只缺少本 PR 的 COW 改动。Reference 数据可作背景，但不能用来证明 COW 本身收益。

### 6.2 合并门禁

必须同时满足：

- 可复用稳态 case 的 allocation/drop 有实测下降，峰值内存如实报告；
- 达到预先声明的主要目标：端到端 optimizer-step 总时间有统计上明确的改善，或峰值内存有明确改善且总时间没有不可接受回退；
- COW 判断成本没有抵消收益；
- 不可复用 case 没有不可接受回退；
- 共享输入绝不被修改；
- 不把 PR 1 相对 Reference 的收益归给本 PR。

若只减少分配次数，但总时间和峰值内存都没有实质改善，保留 out-of-place Fused，不合并本 PR。

## 7. 验证命令

按实际可用 runtime 执行；不可用硬件明确标为 skipped。

```sh
# 默认 Reference 与仓库回归
cargo test -p burn-optim --features burn-core/flex
cargo run-checks

# raw CubeCL CPU 路径与观测
cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu

# Fusion 应继续回退 Reference，不触发专用路径
cargo test -p burn-optim --no-default-features --features std,cpu,burn-core/fusion

# 先验证 benchmark 路径，再采集无观测开销的性能
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu

# GPU runtime 按硬件可用性替换 cpu，并运行对应 raw/Fusion 测试
cargo test -p burn-optim --no-default-features --features std,cuda
cargo run-checks --backend cuda
```

`cargo run-checks --backend ...` 是附加仓库回归，不替代 burn-optim 的 COW/alias 测试和 benchmark。检查后审阅生成 diff，确保 `Cargo.lock` 没有无关变化。

## 8. PR 交付清单与停止条件

PR 描述至少包含：

- 基于的已合并 Fused revision；
- profile 中 allocation/drop 瓶颈证据；
- 复用规则和失败时 out-of-place 行为；
- alias、AMSGrad、checkpoint 测试结果；
- 与 out-of-place Fused 的直接性能表；
- 实测平台和跳过平台；
- 按仓库政策填写 AI Usage。

出现以下任一情况停止本 PR：共享输入必须被修改才能复用、alias 无法证明、COW 检查持续抵消收益，或只有分配次数改善而总时间和峰值内存均无实质收益。停止本 PR 不阻塞其他优化。
