# PR3 本地候选：CPU 模块级 Foreach

状态：**已在本地补丁依赖上实现并验证的候选，尚未接入生产分支；不计 PR3 完成。**

后续[候选修复与分项报告](refinement/README.md)解决占满chunk预算导致的多余参数分配，
补充特殊值、rank8 checkpoint迁移及真实host分项。以下表格保留原candidate4证据，
复现最新候选需在本补丁之后再应用refinement补丁；不得把两版数据混用。

`foreach.patch` 基于 Burn `3221d3f9d`，包含 CPU 多参数 kernel、模块两次遍历、配置、
共同 COW planner、测试及模块 benchmark。依赖仍需要
[execution stream getter](../../prerequisites/README.md)。本目录保存代码和实测证据，
使额外依赖仓库的发布确认到达后可继续集成；它不是另一个独立优化 PR。

生产 Cargo.toml 没有指向本机路径，Cargo 缓存未改动。所有实验源代码和构建均位于本仓库 target 下。
只在本机 root `[patch]` 下通过，不能证明 Burn 作为下游依赖时可构建；发布可获取 revision、
同步 CubeK/Burn pin 和验证下游依赖图仍是交付条件。

## 实现与边界

- `AdamWExecutionConfig::Foreach` 在模块路径尝试收集 **raw CPU F32** 参数；直接单参数调用、
  GPU、超预算输入和 singleton 使用既有 Fused，其他不支持情况沿用 Reference fallback。
- 每项在原执行作用域捕获有效 stream。按 device、client service、有效 stream、dtype、lr/time、
  beta/epsilon/weight decay、state/max 存在性、AMSGrad 和实际 decay mode 分组。
- 一次遍历收集，执行后第二次遍历按 ParamId 写回；保存原 path、优化器组、clipping、lr、time、
  `ParameterContext` 和 ParamMapper。无梯度不推进状态，rank 超过原动态上限仍在原边界失败。
- `Sequence<LinearView>` 为每个张量保留独立 layout。每张量一个编译期 cube 范围 descriptor，
  每 cube 处理最多4096个元素，单位循环处理局部元素；最后 chunk 按逻辑长度检查。
  与原设计逐 chunk 展开相比，代码展开量只随 tensor 数增长。
- 最多8个张量、32个 chunks，另受 runtime 的 binding、cube count 和 cube dimension 限制。
  超预算的单张量在收集前拒绝，累计超限拆分，singleton 在保存的 stream 中走 Fused。
- 单参数与 batch 共用 COW 判定及输出准备。每个输出独立判断独占、连续、allocation 别名；
  batch 还检查同批其他项，包括关闭 AMSGrad 时保留的 max。
- 模块第一遍保留原参数直到第二遍写回，因此 batch 中 parameter buffer 不可独占复用；
  moment 可复用。准备后变成 singleton 的参数也有这个成本。没有用占位 tensor 改变模块语义。
- GPU chunk 候选虽通过数值测试，收益不足以稳定启用。最终候选在模块与 raw 边界均拒绝 GPU batch。

## 正确性与工具检查

| 配置 | 结果 | 实际覆盖范围 |
|---|---|---|
| CPU raw AdamW | 47 passed，2 ignored | 原 Fused/COW/vector 回归及11个新增用例 |
| CPU Foreach JIT，单独运行 | 1 passed | descriptor 不同 key 不同；稳定 layout、不同 lr、新 handles 使用相同 key |
| CPU 原 vector JIT，单独运行 | 1 passed | 原向量/尾部 variant 回归 |
| Metal | 11 passed，1 ignored | 模块、分组、checkpoint、step_multi、rank 及GPU拒绝；CPU专属用例提前返回 |
| native WGPU/WGSL | 11 passed，1 ignored | 同上，实际 native Metal adapter，不代表浏览器WebGPU |
| native WGPU/MSL | 11 passed，1 ignored | 同上 |
| CPU Fusion | 11 passed，1 ignored | 模块Reference回退；raw专属用例提前返回 |
| Flex full burn-optim | 140 passed | 无CubeCL bridge的默认语义 |
| raw CPU all-targets clippy | 通过 | `--no-deps -- -D warnings`；依赖仍输出既有warning |

新增判别用例包含混合rank/transpose、缺梯度time、最后匹配优化器组、不同lr、clipping、
Reference↔Foreach checkpoint、重参数化和两份梯度的step_multi、跨chunk及尾部非恒定值、
累计40个chunk拆成batch+singleton、单项超限拒绝、AMSGrad开关迁移、跨项alias、
关闭AMSGrad时max保留、独占state复用，以及从第三个ambient stream执行后恢复。
GPU/Fusion行的“11 passed”包含按能力提前返回的CPU测试，不能解读为GPU batch通过11项验证。

原始日志摘要和哈希见 [validation.json](validation.json) 与 `validation/`。
Release编译成功，但工具链的`rust-objcopy`找不到`libLLVM.dylib`，去除debug信息失败；
这是编译输出中的warning，binary仍生成并实际运行，未声称strip成功。
本增量未重新运行全仓库`cargo run-checks`、no_std和不可用硬件；既有全仓库阻塞见总体报告。

## CPU 性能：逐参数 Fused 与 Foreach

环境沿用总体报告：Apple M4、macOS 27.0.1、Rust 1.100.0-beta.7。
所有测量通过绝对 cargo/rustc/rustdoc 路径、清空两项rustc wrapper、取消test runner、独立target构建。
没有使用此前已证实存在source/binary不一致的kache构建。

两边是相同候选中的 `ModuleOptimizer`，只改变 execution。每种workload两个独立、初值相同的模型；
每边预热8步，40组配对，组内交替执行顺序，每步前后sync，重复3个进程。
梯度构造在计时区外；step包括模块遍历、state迁移、元数据、分配及提交。
表中耗时是“三轮各自中位数”的中位数，单位ms；加速比列逐轮列出，不能用汇总耗时相除替代配对比值。
`total`包含enqueue和Device::sync，不是纯device时间；两模型同时存活对双方影响对称。

| workload | 参数分布 | Fused enqueue | Fused total | Foreach enqueue | Foreach total | 三轮配对加速比 |
|---|---|---:|---:|---:|---:|---|
| tiny | 30×[64,64] | 0.2037 | 0.9489 | 0.2327 | 0.6406 | 2.054 / 1.456 / 1.147 |
| mixed_rank | [64]+[64,64]+[16,8,3,3] | 0.0233 | 0.1736 | 0.0278 | 0.0734 | 0.971 / 2.380 / 2.367 |
| long_short | tiny+2×[1024,1024] | 0.2267 | 2.1891 | 0.2734 | 1.9351 | 1.129 / 1.411 / 1.000 |
| large_1 | 1×[1024,1024] | 0.0089 | 0.4470 | 0.0102 | 0.4380 | 1.026 / 0.993 / 1.004 |
| large_4 | 4×[1024,1024] | 0.0417 | 2.1805 | 0.0496 | 2.3029 | 0.920 / 0.947 / 1.066 |
| large_16 | 16×[1024,1024] | 0.1476 | 9.3450 | 0.1739 | 9.5165 | 1.038 / 0.982 / 0.730 |
| over_limit | 80×[64,64] | 6.1846 | 7.2247 | 0.5910 | 2.0113 | 2.798 / 4.213 / 3.592 |
| near_limit | 8×[512,256] | 0.0579 | 0.9960 | 0.0713 | 0.9643 | 1.110 / 1.290 / 0.877 |

tiny、over_limit三轮均有净收益；mixed_rank波动较大，long_short最差接近持平。
large_4、large_16包含明显回退样本；它们实际均走逐参数Fused，候选增加了准入检查，且主机负载会影响
同步耗时。当前证据不能证明所有大参数场景无回退，不能将波动一概归为噪声。
near_limit每参数恰好32chunks，无法两项合批，所以仍为8次单参数launch，但多经过收集/写回。

将三轮每模式共120个样本合并后，按nearest-rank计算的total p95如下（ms）。这些尾延迟包含
本机调度与等待，不能用中位数收益推断训练每步都按同一比例加速。

| workload | Fused p95 | Foreach p95 |
|---|---:|---:|
| tiny | 10.2591 | 2.1758 |
| mixed_rank | 1.6760 | 0.0840 |
| long_short | 11.2050 | 3.3107 |
| large_1 | 0.5626 | 0.5780 |
| large_4 | 3.6812 | 3.8154 |
| large_16 | 17.3467 | 15.3023 |
| over_limit | 18.4181 | 3.3515 |
| near_limit | 2.4672 | 2.7530 |

### 路径、首步、内存和遍历成本

下表来自另起进程的固定模式测量，避免两个模型同时存活污染内存峰值；运行时没有本agent的并行编译。
首步包含JIT及执行，仅一次样本，不能当作纯编译耗时。
峰值为各allocator pool峰值之和，是上界，不是某时刻实际总峰值。

| workload | Fused单参数launch | Foreach batch+single | 首步 Fused/Foreach ms | pool峰值 Fused/Foreach MiB |
|---|---:|---|---|---|
| tiny | 30 | 4+0 | 9.523 / 80.723 | 8 / 8 |
| mixed_rank | 3 | 1+0 | 14.992 / 18.274 | 8 / 8 |
| long_short | 32 | 4+2 | 17.394 / 90.574 | 48 / 48 |
| large_1 | 1 | 0+1 | 8.953 / 9.289 | 28 / 28 |
| large_4 | 4 | 0+4 | 14.742 / 12.572 | 88 / 88 |
| large_16 | 16 | 0+16 | 26.817 / 22.022 | 328 / 328 |
| over_limit | 80 | 10+0 | 33.791 / 57.717 | 8 / 8 |
| near_limit | 8 | 0+8 | 9.080 / 10.248 | 24 / 28 |

near_limit的pool峰值增加16.7%，与收集保留parameter导致COW无法复用相符。
tiny首步成本约多71ms，长期训练才能摊销。key专项只证明key区别及稳定性，不直接测量编译次数。

benchmark另测连续100次“两遍identity map”的总时间；每对遍历约tiny 0.00179ms、
mixed_rank 0.00021ms、over_limit 0.00423ms。identity map没有收集、分组、state builder或输出恢复，
只能说明遍历本身的下界，不能称为完整metadata成本。实际相关开销已包含在上表enqueue/total。
最终模型与state的drop+sync在这些CPU样本中约0.006–0.025ms，保留在原始JSON中。

## GPU 实验与启用决策

此前候选2对Metal启用batch：每tensor一个descriptor、每cube256个元素、最多64chunks，
因binding预算tiny稳态为8次batch，over_limit为20次。实测完整模块总耗时比值如下：

| workload | 首轮 Fused/Foreach | 后续三轮配对进程 |
|---|---:|---|
| tiny | 0.737× | 1.114 / 1.113 / 1.079 |
| mixed_rank | 1.016× | 0.993 / 0.989 / 1.037 |
| long_short | 1.006× | 1.010 / 1.020 / 1.037 |
| over_limit | 0.931× | 0.988 / 0.999 / 0.947 |

这组结果不支持全面启用Metal batching，最终候选限定CPU。候选2的kernel、bench、源码hash、
binary hash及原始结果独立保存在`data/gpu-experiment/`，不能与最终CPU-only源码混用。
完整历史候选见该目录`variant.patch`。manifest里的binary路径是测量时路径，后续构建可能覆盖
同名文件；保存的hash是当时测量记录，不能用来声称当前路径仍保存历史binary。
WGPU/MSL、WGPU/WGSL只验证最终回退行为，没有将Metal实验当作这两条runtime的batch性能证据。
CUDA/ROCm硬件、Vulkan SDK和浏览器WebGPU仍不可用；Flex不走CubeCL batch。

最终CPU-only候选另在Metal执行一次同进程40组交替配对。两模式都走单参数Fused，
Fused/Foreach total中位数（ms）为：tiny `0.4978/0.4999`、mixed_rank `0.1742/0.1750`、
long_short `1.1273/1.1252`、large_16 `5.0373/5.0474`、over_limit `0.7100/0.7003`。
比值0.995–1.014，支持该fallback在本次测量中接近持平，不是GPU batching收益。
结果在`data/metal-final-*`，对应最终候选源码hash。

## 复现与后续门禁

从本仓库根目录执行。脚本只读现有CubeCL git checkout，把指定revision展开到本仓库target，
应用getter补丁，并为所有CubeCL workspace包生成同一来源的本地覆盖。现有destination会拒绝覆盖。

```sh
python3 docs/optim-adamw/experiments/pr3-local/prepare.py <existing-cubecl-checkout>
git apply --check docs/optim-adamw/experiments/pr3-local/foreach.patch
git apply docs/optim-adamw/experiments/pr3-local/foreach.patch
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  RUSTC=<beta>/bin/rustc RUSTDOC=<beta>/bin/rustdoc \
  CARGO_TARGET_DIR=target/adamw-foreach-build \
  <beta>/bin/cargo test --config target/adamw-foreach-local.toml \
    -p burn-optim --no-default-features --features std,cpu --lib adamw
# JIT用例分别单独运行，不与其他测试并行。
# vector_width_has_distinct_and_reusable_jit_keys -- --ignored --test-threads=1
# foreach_jit_reuses_stable_layout_and_distinguishes_descriptors -- --ignored

# 使用相同环境构建bench，记录Cargo实际打印的binary路径。
# cargo bench ... --bench adamw --no-run
# <binary> --module-samples paired tiny
# <binary> --module-samples foreach tiny
```

本地`[patch]`不是生产依赖方案。完成远程getter revision、CubeK/Burn同步pin、下游依赖图验证后，
仍需复核性能门禁（尤其大参数回退和metadata分项），完成PR3连续两轮及全分支连续三轮review，
再更新正式PR3/总体报告。本地候选的审查不能提前替代这些交付步骤。

本地候选已由Meitner、Feynman连续两轮agent review通过；第1轮覆盖代码/bench/tests，
第2轮同时核对完整归档、补丁重建、哈希和报告算术。两轮均未重新执行测试，
结果见[审查记录](../../review-log.md)。
