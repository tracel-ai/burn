# PR2：AdamW COW 输出复用报告

2026-11-06，同机 Apple M4 / macOS 27.0.1 / rustc 1.100.0-beta.7，bench optimized。
直接对照是保存的 PR1 out-of-place Fused 二进制；candidate 只增加 COW，仍为标量 kernel。
主要验收目标预先选定为降低峰值内存，同时检查稳态总时间和不可复用场景的代价。
审查门禁记录见 [review-log.md](review-log.md)。

## 结果与实现

- large_16 的池峰值上界：CPU **448→328 MiB（−26.8%）**；Metal **343→328 MiB（−4.4%）**。
  large_1 为 CPU **63→28 MiB（−55.6%）**、Metal **43→28 MiB（−34.9%）**。
- CPU/Metal/WGPU-MSL 的 large_16 独占稳态输出 **48/48** 复用；持有外部克隆时 **0/48** 复用。
  [原始观测](data/pr2-reuse.json) 中的 new_output_ids 是逻辑输出 buffer 的分配，不能解释为底层 malloc 次数。
- 每个 param/m1/m2/max 独立检查 `can_mut`、连续布局及其他角色的 service＋allocation ID。
  只共享 m1 时，其余独立输出仍可复用；同 allocation 的不同 slice 保守视为别名。
- 全部 primitive 转换成功后释放高层 wrapper 的临时引用。若 primitive 拒绝，返回的完整输入
  重建 Reference fallback，恢复旧 time 及 maximum。转换失败仍用原高层输入。
- alias input positions 为 param=0、m1=2、m2=3、old max=4。不存在的 state/max 不绑定 dummy buffer；
  非 AMSGrad 保留的 max 参与冲突检查并原样返回。

## 测量方法

工作负载、超参数、输入数据与 PR1 相同，见 [PR1 报告](pr1-fused-report.md)。每进程预热5步、
采30个 step+sync 样本。先对每个二进制执行 --verify-path，随后性能比较不安装 observer。

`compare.py` 在每个 workload 上执行4轮相邻 A/B，顺序交替为 A→B、B→A、A→B、B→A；每侧每 workload
共120个原始样本。表中的 median 是4个轮次中位数的中位数，speedup 是4个相邻比值的中位数，
p95 来自该侧120个样本。共享对照在每步前持有参数/state克隆，step同步后再释放。

内存与首次执行在独立进程中运行；内存取3步后的活跃用量及各池峰值之和，仍是上界。
首次执行包含编译、提交与执行，不能当成纯 JIT 时间。每次运行均保存二进制及源码 SHA-256。
系统存在桌面应用负载；相邻 A/B 减少时间漂移，但不能消除所有调度噪声。

## 正确性与限制

独占复用、单角色共享、全部角色外部共享、param==grad、m1/m2/max别名、transpose/narrow、
非 AMSGrad 保留旧 max、primitive 拒绝后的恢复均有真实 allocation ID 和数值断言。
narrow 可能被 runtime 物化为连续副本，测试检查真实 metadata。
50步、特殊值、clipping、module和双向checkpoint回归继续执行。

CPU、WGPU/MSL、WGPU/WGSL各35项AdamW测试通过；Metal34项及新增capture测试单独通过。
Metal、WGPU/MSL、WGPU/WGSL上的capture/replay用例验证复用输入全部保活、allocation ID保持，
重放一次与相同标量的eager step一致。CPU无graph实现，明确跳过；未声称验证跨步动态修改lr/time的graph。
Fusion继续回退Reference。CUDA/ROCm无硬件、Vulkan缺SDK、浏览器WebGpu不可用；Flex继续原实现。
no_std及burn-cubecl raw单测的既有编译问题沿用PR1记录，未标记通过。

## 相邻 A/B 数据

主目标的内存下降已成立，独占输入上的 CPU 总时间改善明显。Metal/WGPU 的总时间主要持平或小幅改善，
不将 COW 描述为普遍的大幅 GPU 加速。共享 CPU 首轮 tiny/mixed_rank/large_4 比较偏慢，已额外
执行4轮复测，并完整保留 [复测样本](data/pr2-shared-cpu-repeat.json)。合并8轮后，tiny 的两侧
轮次中位数为18.962/20.253 ms，相邻加速比中位数0.893；mixed_rank 为0.414/0.484 ms、相邻比值1.051，
large_4 为12.705/12.830 ms、相邻比值0.962。比值与两侧中位数不能互相替代。

共享 CPU tiny 存在观测到的约7%中位时间增加（相邻比值反映约12%），不宣称已证明完全无回退。
单独的 enqueue 检查中，mixed_rank 两侧约15–16微秒，large_4约36–40微秒，总时间漂移远大于
可观测的 host 检查开销。当前数据无法把全部波动归因于 COW；采用它的明确代价是这项仍待隔离负载
复测的不确定性。共享 Metal 对照变化约2%以内。此优化保持显式Fused选择，默认Reference不变。

Fusion35项测试与CPU raw all-targets clippy均通过。最终版本连续两轮只读review（Faraday、Hypatia）
均通过，包含新增capture回归；早一轮未包含该回归的PASS不用于最终两轮计数。


### cpu / shared=False

[Raw samples](data/pr2-paired-cpu.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 23.6275 | 2.9446 | 8.109 | 54.5417 | 40.7093 |
| mixed_rank | 0.6424 | 0.1998 | 3.728 | 6.3475 | 5.3928 |
| large_1 | 2.0451 | 1.1359 | 1.835 | 4.2880 | 1.5978 |
| large_4 | 14.6744 | 12.0506 | 1.226 | 27.1823 | 20.1901 |
| large_16 | 59.7571 | 46.8889 | 1.275 | 105.6856 | 95.1713 |
| long_short | 37.1643 | 8.6777 | 4.070 | 68.7164 | 46.2482 |

### metal / shared=False

[Raw samples](data/pr2-paired-metal.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 0.4396 | 0.3989 | 1.095 | 0.7462 | 0.6016 |
| mixed_rank | 0.1275 | 0.1360 | 1.027 | 0.2985 | 0.4212 |
| large_1 | 0.5097 | 0.4781 | 1.068 | 0.8055 | 0.7920 |
| large_4 | 1.5624 | 1.5506 | 1.009 | 2.0940 | 1.8813 |
| large_16 | 5.5628 | 5.5060 | 1.002 | 6.4713 | 6.3976 |
| long_short | 1.1172 | 1.0537 | 1.060 | 1.3311 | 1.2983 |

### wgpu-msl / shared=False

[Raw samples](data/pr2-paired-wgpu-msl.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 0.5860 | 0.5417 | 1.071 | 0.8245 | 0.7777 |
| mixed_rank | 0.3526 | 0.2766 | 1.018 | 0.5222 | 0.5073 |
| large_1 | 0.6380 | 0.6164 | 1.037 | 0.8620 | 0.8358 |
| large_4 | 1.8067 | 1.7532 | 1.032 | 2.4013 | 2.8508 |
| large_16 | 6.0118 | 6.0288 | 0.982 | 6.9057 | 7.5145 |
| long_short | 1.2524 | 1.1997 | 1.043 | 1.5775 | 1.4407 |

### wgpu / shared=False

[Raw samples](data/pr2-paired-wgsl.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 0.5780 | 0.5346 | 1.064 | 0.7840 | 0.7500 |
| mixed_rank | 0.2776 | 0.2720 | 1.018 | 0.3723 | 0.4276 |
| large_1 | 0.6333 | 0.6302 | 0.998 | 0.7458 | 0.8823 |
| large_4 | 1.6772 | 1.6329 | 1.013 | 2.1727 | 2.1648 |
| large_16 | 5.6272 | 5.4608 | 1.031 | 6.3991 | 6.3243 |
| long_short | 1.2694 | 1.1880 | 1.061 | 1.8261 | 1.4955 |

### cpu / shared=True

[Raw samples](data/pr2-shared-cpu.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 19.8689 | 23.1355 | 0.862 | 35.5039 | 61.2278 |
| mixed_rank | 0.4208 | 0.5340 | 0.794 | 3.8239 | 3.4428 |
| large_1 | 1.8149 | 1.8121 | 0.986 | 2.3390 | 2.5123 |
| large_4 | 12.4446 | 14.9299 | 0.855 | 20.3565 | 25.1784 |
| large_16 | 52.9704 | 52.4848 | 1.015 | 79.3915 | 68.2029 |
| long_short | 27.3642 | 28.0929 | 0.969 | 44.1936 | 41.1844 |

### metal / shared=True

[Raw samples](data/pr2-shared-metal.json)

| Workload | PR1 median ms | COW median ms | Speedup | PR1 p95 ms | COW p95 ms |
|---|---:|---:|---:|---:|---:|
| tiny | 0.4228 | 0.4311 | 0.989 | 0.6686 | 0.6510 |
| mixed_rank | 0.1670 | 0.1662 | 0.998 | 0.3745 | 0.3729 |
| large_1 | 0.4987 | 0.5065 | 0.988 | 0.8103 | 0.8065 |
| large_4 | 1.5759 | 1.5302 | 1.043 | 2.2628 | 1.9573 |
| large_16 | 5.7317 | 5.6636 | 1.010 | 6.5229 | 6.1763 |
| long_short | 1.1258 | 1.1255 | 0.992 | 1.6140 | 1.7611 |

## Candidate enqueue, memory and first-step details

### cpu

原始数据：[ pr2-cpu.json ](data/pr2-cpu.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 0.1292 | 4.1035 | 22.2474 | 0.0076 |
| mixed_rank | fused | 0.0157 | 0.1186 | 7.9074 | 0.0117 |
| large_1 | fused | 0.0084 | 0.9478 | 1.3458 | 0.0061 |
| large_4 | fused | 0.0357 | 10.9506 | 19.2995 | 0.0112 |
| large_16 | fused | 0.0949 | 39.9472 | 65.8370 | 0.0085 |
| long_short | fused | 0.1441 | 8.1898 | 30.8411 | 0.0069 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 120 | 1.875 | 8.000 | 13.045 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 18.656 |
| large_1 | fused | 4 | 16.000 | 28.000 | 9.840 |
| large_4 | fused | 16 | 64.000 | 88.000 | 29.337 |
| large_16 | fused | 64 | 256.000 | 328.000 | 47.015 |
| long_short | fused | 128 | 33.875 | 48.000 | 19.458 |

### metal

原始数据：[ pr2-metal.json ](data/pr2-metal.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 0.1079 | 0.5709 | 0.6218 | 0.0411 |
| mixed_rank | fused | 0.0138 | 0.3555 | 0.5180 | 0.0368 |
| large_1 | fused | 0.0061 | 0.4631 | 0.7996 | 0.0527 |
| large_4 | fused | 0.0204 | 1.5235 | 1.7619 | 0.0607 |
| large_16 | fused | 0.0841 | 5.5192 | 12.0866 | 0.0594 |
| long_short | fused | 0.1367 | 1.1701 | 1.9440 | 0.0584 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 120 | 1.875 | 8.000 | 5.528 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 9.250 |
| large_1 | fused | 4 | 16.000 | 28.000 | 5.610 |
| large_4 | fused | 16 | 64.000 | 88.000 | 6.653 |
| large_16 | fused | 64 | 256.000 | 328.000 | 27.067 |
| long_short | fused | 128 | 33.875 | 53.000 | 7.403 |

### wgpu-msl

原始数据：[ pr2-wgpu-msl.json ](data/pr2-wgpu-msl.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 0.1082 | 0.5068 | 0.7562 | 0.0096 |
| mixed_rank | fused | 0.0133 | 0.2853 | 0.4835 | 0.0092 |
| large_1 | fused | 0.0081 | 0.7830 | 1.2683 | 0.0176 |
| large_4 | fused | 0.0210 | 1.7205 | 2.3677 | 0.0166 |
| large_16 | fused | 0.0724 | 5.5750 | 6.3079 | 0.0189 |
| long_short | fused | 0.1203 | 1.1970 | 1.4799 | 0.0125 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 126 | 1.875 | 8.000 | 7.311 |
| mixed_rank | fused | 28 | 0.082 | 8.000 | 9.620 |
| large_1 | fused | 10 | 16.000 | 28.000 | 6.303 |
| large_4 | fused | 22 | 64.000 | 88.000 | 9.532 |
| large_16 | fused | 70 | 256.000 | 328.000 | 14.758 |
| long_short | fused | 139 | 33.876 | 48.000 | 6.756 |

### wgpu

原始数据：[ pr2-wgsl.json ](data/pr2-wgsl.json)；features `std,wgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 0.1077 | 0.4885 | 0.6494 | 0.0103 |
| mixed_rank | fused | 0.0160 | 0.2748 | 0.5042 | 0.0106 |
| large_1 | fused | 0.0071 | 0.6563 | 1.0414 | 0.0141 |
| large_4 | fused | 0.0207 | 1.6856 | 2.4840 | 0.0139 |
| large_16 | fused | 0.0651 | 5.3478 | 5.9848 | 0.0138 |
| long_short | fused | 0.1228 | 1.2636 | 2.2780 | 0.0163 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | fused | 126 | 1.875 | 8.000 | 1.414 |
| mixed_rank | fused | 28 | 0.082 | 8.000 | 2.225 |
| large_1 | fused | 10 | 16.000 | 28.000 | 1.749 |
| large_4 | fused | 22 | 64.000 | 88.000 | 3.912 |
| large_16 | fused | 70 | 256.000 | 328.000 | 10.189 |
| long_short | fused | 139 | 33.876 | 48.000 | 2.555 |

## Reproduction

```sh
python3 docs/optim-adamw/compare.py <pr1-binary> <pr2-binary> paired.json \
  --backend metal --features std,cpu,metal,wgpu-msl,webgpu
# Each binary requires a matching <binary>.json source manifest.
# Add --shared for the externally retained input control.
BURN_DEVICE=metal <pr2-binary> --measure-reuse fused large_16
BURN_DEVICE=metal <pr2-binary> --measure-reuse fused large_16 --shared
```

Implementation, measurement, documentation and reviews were performed by coding agents. No human review is claimed.
