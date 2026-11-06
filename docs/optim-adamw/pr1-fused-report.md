# PR1：AdamW 单参数 Fused 优化报告

测量日期：2026-11-06。Reference 与 Fused 在相同二进制、相同设备和 workload 内比较。
本报告使用修正后的预热、同步与内存测量流程。连续 review 状态以 [review-log.md](review-log.md) 为准。

## 实现与结果

每参数一个 kernel 更新参数、一阶矩、二阶矩及可选 AMSGrad maximum。F32 raw CubeCL 支持
普通与 cautious decay；默认仍为 Reference。Fusion、autodiff 直接调用、非 F32、空张量、
rank 0、tiled/量化、不匹配的 shape/device/service、无效 buffer span 或 binding 预算不足时回退。
COW 与向量化尚未加入本次对照。

| 后端 | workload | Reference median ms | Fused median ms | 加速比 |
|---|---|---:|---:|---:|
| cpu | tiny | 233.0470 | 13.9834 | 16.67× |
| cpu | mixed_rank | 1.2098 | 0.0807 | 14.99× |
| cpu | large_1 | 11.9676 | 2.1356 | 5.60× |
| cpu | large_4 | 42.3351 | 15.4470 | 2.74× |
| cpu | large_16 | 260.8188 | 58.1493 | 4.49× |
| cpu | long_short | 275.6952 | 32.1388 | 8.58× |
| metal | tiny | 2.0080 | 0.6720 | 2.99× |
| metal | mixed_rank | 0.5259 | 0.1632 | 3.22× |
| metal | large_1 | 1.5777 | 0.5960 | 2.65× |
| metal | large_4 | 6.0626 | 1.7430 | 3.48× |
| metal | large_16 | 24.5775 | 6.1523 | 3.99× |
| metal | long_short | 4.1615 | 1.1820 | 3.52× |
| wgpu-msl | tiny | 2.8100 | 0.8915 | 3.15× |
| wgpu-msl | mixed_rank | 0.4928 | 0.4448 | 1.11× |
| wgpu-msl | large_1 | 1.8505 | 0.6675 | 2.77× |
| wgpu-msl | large_4 | 6.3896 | 2.2218 | 2.88× |
| wgpu-msl | large_16 | 22.9257 | 5.8857 | 3.90× |
| wgpu-msl | long_short | 4.4309 | 1.3451 | 3.29× |
| wgpu | tiny | 1.4096 | 0.7298 | 1.93× |
| wgpu | mixed_rank | 0.5589 | 0.3051 | 1.83× |
| wgpu | large_1 | 1.7836 | 0.6553 | 2.72× |
| wgpu | large_4 | 6.1904 | 1.7811 | 3.48× |
| wgpu | large_16 | 23.5059 | 5.9305 | 3.96× |
| wgpu | long_short | 4.3817 | 1.3313 | 3.29× |

机器为 Apple M4（10 CPU cores、10 GPU cores、16 GiB），macOS 27.0.1，rustc 1.100.0-beta.7，
bench optimized。系统同时有桌面应用负载，尤其 CPU runtime 的调度噪声明显；以下 p95 和原始
样本均保留。单轮中位数证明这次运行的收益，不把它当作隔离机器上的稳定极限。

## 工作负载和计时边界

| workload | 参数分布 | 元素总数 |
|---|---|---:|
| tiny | 30 × [64,64] | 122880 |
| mixed_rank | [64] + [64,64] + [16,8,3,3] | 5312 |
| large_1 | [1024,1024] | 1048576 |
| large_4 | 4 × [1024,1024] | 4194304 |
| large_16 | 16 × [1024,1024] | 16777216 |
| long_short | 30 × [64,64] + 2 × [1024,1024] | 2220032 |

`lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-5, decay=1e-4`；非 AMSGrad、非 cautious。
参数从 ones 开始，梯度固定 0.01，state 跨步传递；每次测量更新全部参数。

- 每个 workload/mode 独立进程，先在待测 case 上预热 5 步，再采 30 次。
- `enqueue_ms` 包含整步 host 提交及可能的反压；`total_ms` 包含随后 `Device::sync`。
  每次采样前一轮队列已排空。p95 使用 nearest-rank（第 29/30 个有序样本）。
- retained drop+sync 是释放额外共享引用后的同步时间；本表独占模式没有额外引用，因此该列
  主要是空同步开销。step 内旧输入释放包含在 enqueue 中；最终全部 case 的释放时间单独记录于 JSON。
- 首步时间是 setup 并同步后的第一次 optimizer step，包含冷 kernel 编译与执行；不是纯 JIT 编译时间，
  不承诺驱动缓存完全冷启动。未用首次执行数值代替稳态。
- 内存在另一个独立进程中运行 3 步后读取，测量进程不做克隆的路径 proof。`allocs` 是当前活跃
  分配数，不是累计 allocation 次数。`peak` 是每个池的峰值页数乘页大小之和，是总峰值上界。
- `--verify-path` 独立比较同一组随机输入上的 3 步 param/m1/m2，并用现有 CubeCL observer
  确认 Initial/State 均提交实际 AdamW kernel。测试另覆盖 AMSGrad、cautious、布局与回退。

## 数值与结构

Reference 的普通稳态路径有 15 个高阶 tensor 算子，Fused 一次提交；这是源码结构计数，
不把“15 趟”误写成精确显存流量或所有模式的物理 kernel 数。Fused 读 p/g/m1/m2、写 p/m1/m2，
AMSGrad 增加 maximum 的读写；PR1 为输出分配 3/4 个新 buffer。

所有与后续加减相邻的乘法使用 `fma(a,b,negative_zero)`，负零为 runtime scalar，保留
Reference 独立乘法的舍入、溢出及 signed-zero；字面量负零会被 WGSL 编译链折叠，不能作为屏障。
数值容差 `rtol=8ε, atol=4ε`；Inf/±0 位模式和 NaN 分类单独检查。

## 逐后端原始统计

### cpu

原始数据：[ pr1-native-cpu.json ](data/pr1-native-cpu.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 226.4416 | 233.0470 | 592.3448 | 0.0092 |
| tiny | fused | 12.3641 | 13.9834 | 28.4248 | 0.0083 |
| mixed_rank | reference | 0.0846 | 1.2098 | 7.2247 | 0.0078 |
| mixed_rank | fused | 0.0163 | 0.0807 | 0.6703 | 0.0071 |
| large_1 | reference | 0.0710 | 11.9676 | 48.3934 | 0.0107 |
| large_1 | fused | 0.0222 | 2.1356 | 2.7021 | 0.0073 |
| large_4 | reference | 15.5739 | 42.3351 | 101.3642 | 0.0156 |
| large_4 | fused | 0.0526 | 15.4470 | 25.8633 | 0.0122 |
| large_16 | reference | 208.0959 | 260.8188 | 517.3500 | 0.0123 |
| large_16 | fused | 0.1161 | 58.1493 | 125.3125 | 0.0070 |
| long_short | reference | 251.5106 | 275.6952 | 642.8556 | 0.0119 |
| long_short | fused | 16.0398 | 32.1388 | 54.9048 | 0.0069 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 120 | 1.875 | 8.000 | 122.799 |
| tiny | fused | 120 | 1.875 | 8.000 | 12.659 |
| mixed_rank | reference | 12 | 0.081 | 8.000 | 261.556 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 14.688 |
| large_1 | reference | 4 | 16.000 | 48.000 | 67.105 |
| large_1 | fused | 4 | 16.000 | 63.000 | 12.403 |
| large_4 | reference | 16 | 64.000 | 108.000 | 114.171 |
| large_4 | fused | 16 | 64.000 | 163.000 | 23.584 |
| large_16 | reference | 64 | 256.000 | 348.000 | 270.042 |
| large_16 | fused | 64 | 256.000 | 448.000 | 64.970 |
| long_short | reference | 128 | 33.875 | 68.000 | 163.343 |
| long_short | fused | 128 | 33.875 | 63.000 | 36.074 |

### metal

原始数据：[ pr1-native-metal.json ](data/pr1-native-metal.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 1.5762 | 2.0080 | 2.6518 | 0.0512 |
| tiny | fused | 0.3066 | 0.6720 | 1.1336 | 0.0505 |
| mixed_rank | reference | 0.0931 | 0.5259 | 0.7528 | 0.0536 |
| mixed_rank | fused | 0.0150 | 0.1632 | 0.3800 | 0.0365 |
| large_1 | reference | 0.0307 | 1.5777 | 1.9806 | 0.0493 |
| large_1 | fused | 0.0106 | 0.5960 | 1.4152 | 0.0576 |
| large_4 | reference | 0.1771 | 6.0626 | 6.5136 | 0.0606 |
| large_4 | fused | 0.0326 | 1.7430 | 2.0281 | 0.0614 |
| large_16 | reference | 0.9499 | 24.5775 | 25.3178 | 0.0660 |
| large_16 | fused | 0.1168 | 6.1523 | 6.8546 | 0.0685 |
| long_short | reference | 1.0138 | 4.1615 | 5.4444 | 0.0558 |
| long_short | fused | 0.1695 | 1.1820 | 2.3078 | 0.0648 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 120 | 1.875 | 8.000 | 119.129 |
| tiny | fused | 120 | 1.875 | 8.000 | 32.462 |
| mixed_rank | reference | 12 | 0.081 | 8.000 | 187.586 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 35.117 |
| large_1 | reference | 4 | 16.000 | 38.000 | 79.346 |
| large_1 | fused | 4 | 16.000 | 43.000 | 6.937 |
| large_4 | reference | 16 | 64.000 | 98.000 | 59.476 |
| large_4 | fused | 16 | 64.000 | 103.000 | 8.046 |
| large_16 | reference | 64 | 256.000 | 338.000 | 84.023 |
| large_16 | fused | 64 | 256.000 | 343.000 | 13.428 |
| long_short | reference | 128 | 33.875 | 58.000 | 49.334 |
| long_short | fused | 128 | 33.875 | 63.000 | 7.462 |

### wgpu-msl

原始数据：[ pr1-native-wgpu-msl.json ](data/pr1-native-wgpu-msl.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 1.9162 | 2.8100 | 13.3870 | 0.0358 |
| tiny | fused | 0.1664 | 0.8915 | 7.0115 | 0.0156 |
| mixed_rank | reference | 0.0751 | 0.4928 | 0.8275 | 0.0104 |
| mixed_rank | fused | 0.0160 | 0.4448 | 0.5428 | 0.0110 |
| large_1 | reference | 0.0291 | 1.8505 | 2.2385 | 0.0130 |
| large_1 | fused | 0.0070 | 0.6675 | 0.9160 | 0.0114 |
| large_4 | reference | 0.1279 | 6.3896 | 6.8898 | 0.0175 |
| large_4 | fused | 0.0499 | 2.2218 | 3.7664 | 0.0412 |
| large_16 | reference | 0.8555 | 22.9257 | 28.4447 | 0.0212 |
| large_16 | fused | 0.1004 | 5.8857 | 6.5580 | 0.0162 |
| long_short | reference | 1.1123 | 4.4309 | 4.7828 | 0.0138 |
| long_short | fused | 0.1544 | 1.3451 | 1.6298 | 0.0120 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 138 | 1.875 | 8.000 | 46.243 |
| tiny | fused | 126 | 1.875 | 8.000 | 7.849 |
| mixed_rank | reference | 64 | 0.082 | 8.000 | 142.389 |
| mixed_rank | fused | 28 | 0.082 | 8.000 | 31.185 |
| large_1 | reference | 22 | 16.000 | 38.000 | 43.867 |
| large_1 | fused | 10 | 16.000 | 43.000 | 6.681 |
| large_4 | reference | 34 | 64.000 | 98.000 | 68.032 |
| large_4 | fused | 22 | 64.000 | 103.000 | 8.707 |
| large_16 | reference | 82 | 256.000 | 338.000 | 60.225 |
| large_16 | fused | 70 | 256.000 | 343.000 | 16.324 |
| long_short | reference | 163 | 33.876 | 58.000 | 43.301 |
| long_short | fused | 139 | 33.876 | 63.000 | 7.501 |

### wgpu

原始数据：[ pr1-wgsl.json ](data/pr1-wgsl.json)；features `std,wgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 1.0591 | 1.4096 | 1.8359 | 0.0121 |
| tiny | fused | 0.1496 | 0.7298 | 1.0738 | 0.0120 |
| mixed_rank | reference | 0.0801 | 0.5589 | 1.3432 | 0.0153 |
| mixed_rank | fused | 0.0149 | 0.3051 | 0.5807 | 0.0083 |
| large_1 | reference | 0.0314 | 1.7836 | 2.4775 | 0.0141 |
| large_1 | fused | 0.0074 | 0.6553 | 1.4429 | 0.0117 |
| large_4 | reference | 0.0991 | 6.1904 | 7.2079 | 0.0122 |
| large_4 | fused | 0.0255 | 1.7811 | 2.4836 | 0.0148 |
| large_16 | reference | 0.6960 | 23.5059 | 24.9865 | 0.0161 |
| large_16 | fused | 0.0817 | 5.9305 | 6.6925 | 0.0149 |
| long_short | reference | 1.1302 | 4.3817 | 5.4018 | 0.0134 |
| long_short | fused | 0.1567 | 1.3313 | 2.0804 | 0.0125 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 138 | 1.875 | 8.000 | 5.725 |
| tiny | fused | 126 | 1.875 | 8.000 | 1.539 |
| mixed_rank | reference | 64 | 0.082 | 8.000 | 12.468 |
| mixed_rank | fused | 28 | 0.082 | 8.000 | 2.339 |
| large_1 | reference | 22 | 16.000 | 38.000 | 5.763 |
| large_1 | fused | 10 | 16.000 | 43.000 | 2.108 |
| large_4 | reference | 34 | 64.000 | 98.000 | 13.504 |
| large_4 | fused | 22 | 64.000 | 103.000 | 3.765 |
| large_16 | reference | 82 | 256.000 | 338.000 | 26.705 |
| large_16 | fused | 70 | 256.000 | 343.000 | 10.351 |
| long_short | reference | 163 | 33.876 | 58.000 | 8.588 |
| long_short | fused | 139 | 33.876 | 63.000 | 2.897 |

## Flex 与不可用平台

Flex 不实现 CubeCL kernel，继续使用 Reference。已有同机数据 [flex.json](data/flex.json)
提供参考量级；它是较早独立运行，源码哈希不同，不参与当前 Fused 加速比计算。CUDA/ROCm 无对应硬件。
Vulkan 构建因缺少 macOS Vulkan SDK 失败，日志见 [Vulkan](data/pr1-vulkan-unavailable.txt)。
`BURN_DEVICE=webgpu` 明确选择浏览器 WebGpu API，在这个原生进程不可用，日志见
[WebGpu](data/pr1-webgpu-unavailable.txt)；启用 `webgpu` feature 而不设置该设备可能仍运行 Metal
适配器，不能据此声称验证了浏览器 WebGPU。

## 复现与数据归属

```sh
cargo bench -p burn-optim --bench adamw --no-default-features \
  --features std,cpu,metal,wgpu-msl,webgpu --no-run
# 使用 Cargo 输出的 executable 路径；依次替换 cpu、metal、wgpu-msl
python3 docs/optim-adamw/measure.py <binary> data.json --backend cpu \
  --features std,cpu,metal,wgpu-msl,webgpu
# 独立 WGSL 构建：不启用 wgpu-msl
cargo bench -p burn-optim --bench adamw --no-default-features --features std,wgpu --no-run
python3 docs/optim-adamw/measure.py <wgsl-binary> data-wgsl.json --backend wgpu --features std,wgpu
```

JSON 保留完整 stdout/stderr、二进制 SHA-256、源码 SHA-256、features、日期和所有样本。
后续 PR2/PR4 使用保存的前一阶段二进制与 source manifest 做直接 A/B；测试代码后续增加不会
修改已记录哈希。早期 `cpu*.json`、`metal.json` 和旧报告中的数字保留作历史数据，不与上述样本混合。
本报告及实现由编码 agent 生成、测试和审查；没有声称维护者或人工已完成 review。

## 验证结果

CPU raw、Metal raw、WGPU/MSL、WGPU/WGSL、CPU Fusion 的 AdamW 测试各 29 项通过。
CPU raw all-targets clippy 通过（依赖中已有警告仍会显示）。验证命令和结果摘要见
[pr1-validation.json](data/pr1-validation.json)。Metal/MSL 验证先在临时快照执行，相关生产源码
及回归测试已完整合回当前仓库；随后修改的 binding 预算测试预期由当前 CPU/WGSL/Fusion 运行覆盖。

`cargo check -p burn-optim --no-default-features` 失败：未改动的 ModuleOptimizer 无条件引用
std 门控的 lr_scheduler，且未导入 alloc::vec/String，共六个编译错误。与 HEAD 源码核对相同，
未把 no_std 标记为通过。CUDA/ROCm、Vulkan、浏览器 WebGpu、graph capture 尚未验证。

补充检查：`cargo test -p burn-cubecl --no-default-features --features std,cpu --lib binding_budget_accounts_for_state_variants`
在未改动的 `#[cfg(any(feature="fusion", test))] mod fusion` 处编译失败：raw 测试未启用
可选 burn-fusion 依赖，产生 unresolved imports 等错误。因此该 crate 内的纯预算单测尚未运行；
burn-optim 各后端实际路径、数值与 budget 相关预期已经运行。此限制不记为测试通过。
