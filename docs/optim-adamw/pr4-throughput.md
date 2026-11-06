# PR 4：AdamW 大参数吞吐优化

> 文档状态：CPU 向量化已实现并通过性能验收；GPU runtime 保留 scalar；完整数据见 [pr4-throughput-report.md](pr4-throughput-report.md)。
>
> 性能类型：提高大参数 Fused 内核的有效吞吐，并在有收益时扩大快路径范围
>
> 代码依赖：单参数 Fused 已合并；不依赖 COW 或 Foreach
>
> 直接性能对照：当前 `main` 上相同 workload 的实际执行路径，可能是标量 Fused，也可能是 PR 1 保守回退的 Reference

本文独立说明该 PR 的触发条件、实现、正确性和性能门禁，不需要阅读其他 PR 文档。

## 1. 性能问题与独立价值

单参数标量 Fused 主要减少调度和中间 tensor，不保证大参数场景达到设备带宽能力。PR 1 为避免回退，可以把部分大参数保守留在 Reference；这不等于已经解决大参数吞吐问题。

如果 profile 证明大参数瓶颈来自标量内存访问、地址计算或 launch 配置，本 PR 针对该瓶颈做一项聚焦的内核优化。

本 PR 的独立收益是：

- 提高已经启用 Fused 的大参数吞吐；或
- 让 PR 1 中因速度不足而回退的大参数，在优化后以获得净收益的方式进入 Fused。

本 PR 不是独立的阈值调参 PR。初始快路径保护阈值由 PR 1 负责；本 PR 只有在内核本身变快后，才可以重新评估并扩大适用范围。

局部 profiler 指标改善但端到端 step 没有收益时，不合并。如果现有 Fused 已在目标大参数场景达到合理吞吐，且没有因性能原因被排除的大参数，本 PR 不需要存在。

## 2. 范围与非目标

本 PR 可以包含：

- 对齐向量加载/存储及标量尾部；
- 或地址计算、launch 配置中的一项聚焦改进；
- 因内核变快而重新评估原有 Fused 准入范围；
- 对应正确性、路径观测和大参数 benchmark。

本 PR 不包含：

- 不改变内核、只调整阈值的调参；
- COW 所有权规划；
- 模块级 Foreach 收集；
- F16/BF16/F64 或量化支持；
- 改变 AdamW 公式、decay、clipping、state 或 checkpoint；
- 未测收益的多常量调优。

如果 Foreach 已合并，不能自动把单参数优化结论扩展到多参数 chunk kernel。只有分别证明单参数和多参数都受益时才同时修改；否则只改单参数路径，多参数版本另立 PR。

## 3. 数值与布局契约

优化只改变每元素计算的执行方式，不改变数学顺序。Host 对每个参数按现有 Rust 精度计算：

```text
time_i32   = state.time as i32
factor_1   = 1.0f32 - beta_1
factor_2   = 1.0f32 - beta_2
bias_1     = 1.0f32 - beta_1.powi(time_i32)
bias_2     = 1.0f32 - beta_2.powi(time_i32)
decay_rate = lr * (weight_decay as f64)
```

对 tensor dtype `F`：

```text
m1 = old_m1 * F(beta_1) + g * F(factor_1)
m2 = old_m2 * F(beta_2) + (g * g) * F(factor_2)

if amsgrad:
    previous = has_old_max ? old_max : m2
    v = previous < m2 ? m2 : previous
else:
    v = m2

m1_corrected = m1 / F(bias_1)
v_corrected = v / F(bias_2)
delta = m1_corrected / (sqrt(v_corrected) + F(epsilon))
```

首步不读取 old state：`m1=g*factor_1`、`m2=(g*g)*factor_2`，AMSGrad 首步以新 m2 初始化 max。已有 state 但无 old max 时，同样以本次新 m2 初始化；非 AMSGrad 保留已有 old max，但不更新、不用于 delta。

decay 保持：

```text
NoDecay:  decayed = p
Standard: decayed = p * F(1.0f64 - decay_rate)
Cautious: decayed = p - (((p >= 0) == (m1 >= 0)) ? p * F(decay_rate) : F(0))
p_new = decayed - delta * F(lr)
```

约束：

- 不做跨元素归约、向量重排或代数重组；
- 不主动用 `mul_add` 改变 Reference 运算边界；
- bias 是分母，不改用 reciprocal 乘法；
- NoDecay 不引入 `0*Inf`，Cautious 不改成 `(p*m1)>=0`，使用更新后的 m1；
- AMSGrad 保持 `old_max < m2` 的选择方向：old NaN 保留、new NaN 不覆盖有限 old、相等/signed zero 保留 old 位模式；
- 参数、全部 state、time、shape、rank、dtype、device 和 checkpoint 名称不变；
- 首版仍只支持 F32 非量化输入，其他 dtype、空张量和 tiled 输入回退 Reference；
- 非连续输入按 LinearView 正确寻址，不因向量化强制 reshape 或 D2H sync；
- F32 是容差兼容而非 bitwise 承诺，特殊值另做分类或位模式断言。

## 4. 实现方案

从以下方向中选择由 profile 支持的一项主要改动，不在一个 PR 中堆叠多种调优。

### 4.1 对齐向量化与标量尾部

适用于标量内存访问限制大参数吞吐的情况：

1. Host 根据所有输入/输出的地址对齐、layout 和 dtype 选择 vector width；
2. 不对齐或 layout 不合格时使用合法的较小宽度或标量 variant；
3. 大连续输入的主体按 vector width 处理，余下元素使用同一 kernel 的 width=1 variant，offset 为主体元素数；不额外分配输出；
4. 所有 param/grad/state read 和 output write 使用同一逻辑 index；
5. JIT key 区分 vector width 和 variant；
6. 不改变 optional max 的读取和写入条件。

vector width 不照搬其他框架常量，按 runtime 实测选择。输出复用若已存在，仍须先通过既有 COW 规则；本 PR 不放宽 alias 条件。

本次未保留的实验使用 runtime 的 `io_optimized_vector_sizes`。全部参与 view 连续时按总元素数处理，
无需限制最内维整除；其他布局使用已有 `max_vector_size` 并取共同宽度。每个 handle 的 offset
进一步限制宽度。若余数非零且向量主体不足一个硬件允许的完整 workgroup，则使用一次标量提交，
避免小不规则张量的额外尾部 launch。较大余数场景最多两次提交，两次均保留所有输入，写入区间不相交。
offset 为运行时索引，不把张量长度放入 JIT key；宽度、布局、alias及已有variant仍属于编译参数。

### 4.2 地址计算或 launch 配置

适用于 profile 明确显示整数地址计算、占用率或 workgroup 配置限制的情况：

1. 只修改一个主要因素，保留可复现对照；
2. 保持 LinearView 的逻辑索引和边界检查语义；
3. 不通过读取越界、取消尾部检查或假定连续布局换取速度；
4. 将 variant、地址类型和 launch 参数纳入正确 JIT key；
5. 按 runtime 分别验证，不能从单一设备外推最优配置。

### 4.3 内核变快后的准入范围

PR 1 可以已有保守的 numel/layout 快路径限制。本 PR 只有在优化后的内核于原回退区间也快于当前实际路径时，才扩大准入：

1. 对照组是当前 main 在该 workload 上真正执行的 Reference 或标量 Fused；
2. 路径观测证明 candidate 实际进入新内核；
3. 回退仍只发生在 launch 前；
4. 不用单一 shape 的交叉点外推整个区间；
5. 如果扩大准入没有额外净收益，保留原范围，本 PR 只声明已优化区间的收益。

## 5. 正确性测试

先证明 candidate 实际进入预期标量/向量 variant，再比较结果。至少覆盖：

| 类别 | 用例与断言 |
|---|---|
| 对齐 | 所有输入输出满足对齐时进入目标 variant |
| 非对齐 | 任一输入不对齐时走合法 variant，不越界、不读错地址 |
| 尾部 | `numel = vector_width*k ± 1`，逐元素验证恰好更新一次 |
| shape | rank 0/1/2/4、含 1 维、大矩阵和长短混合 |
| layout | transpose/narrow 正确寻址，tiled 回退 |
| state | 首步、第二步、50 steps；param/m1/m2/max 全部比较 |
| AMSGrad | 有/无 old max、NaN、signed zero，选择规则不变 |
| decay | NoDecay+Inf、Standard、Cautious mask |
| dtype | 非 F32 继续 Reference，专用计数不增加 |
| JIT | 不同 vector width/variant/launch 配置的 key 不碰撞 |
| 准入范围 | 原回退区间只有在新路径实际更快时才启用，路径计数符合预期 |
| checkpoint | 优化前后 state 可双向恢复，rank/time/名称不变 |

Reference 和 candidate 在同一 device、同一 dtype 下生成，避免跨后端差异。

## 6. Benchmark 与性能验收

### 6.1 测量方法

至少包含：

- 1、4、16 个 `[1024,1024]` F32 参数；
- PR 1 中仍走 Reference 的大参数区间（如存在）；
- 30 个 `[64,64]` 小参数，用于检测小模型回退；
- 长短 tensor 混合；
- `vector_width*k ± 1` 的尾部 shape；
- 拟扩大准入范围时，覆盖边界附近多组 shape。

分别报告 Host enqueue、enqueue+sync 总时间、drop/sync、首次 JIT、稳态 median/p95、有效带宽或元素吞吐、实际 variant/launch 选择和原始样本。记录 hardware、OS、runtime、Cargo features、编译模式、warmup 和样本数。

直接 A/B 只改变本 PR。若 COW 或 Foreach 已存在，A/B 两边必须保持相同配置，不能把其累计收益归给本 PR。对于 PR 1 已回退 Reference 的大参数，必须明确对照组是 Reference，不能仍声称相对标量 Fused 加速。

### 6.2 合并门禁

- 预先声明的大参数目标区间，其端到端吞吐或总时间有统计上明确的改善；
- 小参数 launch-bound workload 没有不可接受回退；
- 首次 JIT、variant 数量和代码体积变化单独报告；
- 向量化尾部、非对齐和布局 case 全部通过；
- 扩大准入范围时，边界附近多组 shape 均有数据支持；
- 性能声明只覆盖实测 runtime 和 workload。

局部指标改善但总时间恶化、收益只存在于单个精心挑选 shape，或必须放宽数值契约才能变快时，不合并。

## 7. 验证命令

按实际实现选择对应 variant、尾部、布局和准入测试，不可用平台明确标为 skipped。

```sh
cargo test -p burn-optim --features burn-core/flex
cargo run-checks

cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu,burn-core/fusion

cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu

# 大参数结论应尽量在实际目标 GPU 上验证；无硬件时明确 skipped
cargo test -p burn-optim --no-default-features --features std,cuda
cargo run-checks --backend cuda
```

`cargo run-checks --backend ...` 不替代 burn-optim 的 variant、尾部、准入路径和性能验证。检查后审阅生成 diff，确保没有无关 `Cargo.lock` 变化。

## 8. PR 交付清单与停止条件

PR 描述至少包含：

- profile 定位的大参数瓶颈；
- 选择向量化、地址计算或 launch 优化的理由；
- 未纳入本 PR 的其他调优；
- variant/尾部/layout/JIT 测试；
- 大参数与小参数直接 A/B，明确各对照组的实际路径；
- JIT、吞吐、总时间、准入变化和限制；
- 实测/跳过平台和 AI Usage。

如果没有独立大参数瓶颈、CPU/目标 runtime 没有端到端收益，或扩大准入后的回退抵消总体价值，停止对应 runtime 的向量化，不影响其他优化。
