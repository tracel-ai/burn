# PR4：AdamW 大参数吞吐报告

状态：**CPU 向量化与 serial scalar tail 已实现；Metal/WGPU/MSL 保留 scalar variant。** 本报告只使用禁用 kache wrapper、绝对 beta cargo/rustc、独立 target 和匹配 source manifest 的 direct binaries。早期缓存构建数据全部标记 invalid，不参与结论。

## 结果

标量基线源码为 PR2 scalar COW；candidate 为当前 CPU-only vector body + serial scalar tail 实现。每个 workload 使用四轮交替 A/B、B/A、A/B、B/A，每侧每轮30样本。每次 path proof 覆盖8个 workload、共输出8行 verified，并保存 binary/source hash。

| Backend | large_1 | large_4 | large_16 | tail_minus | tail_plus |
|---|---:|---:|---:|---:|---:|
| CubeCL CPU | **2.634×** | **2.160×** | **1.893×** | **2.123×** | **2.469×** |
| Metal | 1.020× | 0.995× | 0.994× | 1.035× | 0.954× |
| WGPU/MSL | 1.011× | 1.029× | 1.014× | 0.997× | 1.018× |

比值大于1表示 candidate 更快。CPU large_1/4/16 均达到明确端到端收益，两个尾部 workload 也恢复正收益；之前 tail_minus 的退化来自为每个尾部线程启动一个 CPU unit。修复后 tail 在一个 unit 内 serial loop。GPU 结果接近噪声，保持 scalar path，避免把 CPU收益外推到GPU。

数据文件：

- [CPU serial-tail ABBA](data/pr4-serial-tail-cpu.json)
- [Metal serial-tail ABBA](data/pr4-serial-tail-metal.json)
- [WGPU/MSL serial-tail ABBA](data/pr4-serial-tail-msl.json)
- [当前 binary/source manifest](data/pr4-serial-tail-sources.json)
- [当前 serial-tail path proof](data/pr4-current-path-proof.json)
- [标量基线与原始向量实验的 path proof](data/pr4-direct-path-proof.json)

早期 `pr4-verified-*`、`pr4-paired-*` 是缓存构建归属不可靠的数据，均标记 `valid_for_acceptance=false`。`pr4-direct-*` 中的历史candidate来自禁用wrapper的直接构建，其身份已核实，但被最终serial-tail替代，不用于当前候选验收。`pr4-direct-scalar-sources.json`是例外：它仍是当前A/B使用的有效标量基线，并未被替代。当前验收只用 `pr4-serial-tail-{cpu,metal,msl}.json`，以及同源码的 `pr4-production-*`。

## 实现

CPU runtime 根据所有输入/输出共同布局和 `io_optimized_vector_sizes` 选择宽度；不连续、偏移不对齐或不适合向量化时降到合法较小宽度。大连续 tensor 使用向量主体，剩余元素以 width=1 serial tail 处理。tail仅启动一个unit，并在kernel中拒绝非零unit；循环范围为 `0..units`，host保证 `offset+units=numel`，所以索引加法不会回绕到已更新的开头。所有非CPU runtime强制width=1，Metal/MSL/WGSL验证该路径，未外推CUDA/ROCm实测。

COW planner未放宽，alias位置仍为param=0、m1=2、m2=3、max=4。主体和tail提交之间保持所有输入和输出生命周期，两个写区间不重叠。JIT回归验证vector和scalar variant key区分，runtime lr不改key。

实验/最终差异见 [pr4-vectorized.patch](experiments/pr4-vectorized.patch)，source hash见 [pr4-source-hashes.json](experiments/pr4-source-hashes.json)。当前 patch 可在 `target/adamw-pr2-source` scalar snapshot 中通过 `git apply --check`；当前工作树已包含该实现，不能直接重复应用。

最终提交前仅由rustfmt将kernel一个if表达式换行；测量源码hash与最终格式化hash的映射记录在
[final-formatting.json](data/final-formatting.json)，去除空白后的源码一致。原始binary/source manifest保持原值。

## 验证

当前CPU/Metal/WGPU-MSL/WGPU-WGSL/CPU-Fusion AdamW测试各36项通过，CPU JIT key测试单独通过；CPU odd lengths、offset、WGPU storage alignment、transpose/narrow、AMSGrad、cautious、overflow、COW、capture和checkpoint覆盖。当前测试及raw CPU all-targets clippy使用禁用wrapper的独立target并通过，验证记录见[pr4-current-validation.json](data/pr4-current-validation.json)。Vulkan缺SDK，CUDA/ROCm无硬件，浏览器WebGPU不可用。

已知边界：CPU结论只覆盖Apple M4 CubeCL CPU runtime，GPU scalar结论只覆盖本机Metal/WGPU-MSL。约16GiB U32极限buffer仅做源码算术核验，未实机分配。

PR4最终版本已由Darwin、Sagan连续两轮独立review通过；raw CPU all-targets clippy亦通过。性能门槛由direct source-matched数据支持CPU目标；GPU保留scalar，测量限制如上。

## 完整稳态对照

每个单元格为四轮中位数的中位数；p95取每侧120个样本的nearest-rank。比值为配对轮次比值中位数，不能用表内两个中位数相除替代。

### cpu

| workload | scalar total median ms | candidate total median ms | speed ratio | scalar p95 ms | candidate p95 ms | scalar enqueue median ms | candidate enqueue median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 1.7837 | 0.8757 | 1.950× | 12.6307 | 6.5152 | 0.1200 | 0.1519 |
| mixed_rank | 0.2297 | 0.1548 | 1.490× | 1.7732 | 2.5458 | 0.0147 | 0.0182 |
| large_1 | 1.2606 | 0.5111 | 2.634× | 1.8514 | 0.6435 | 0.0101 | 0.0084 |
| large_4 | 9.8912 | 4.3687 | 2.160× | 16.4786 | 15.5252 | 0.0347 | 0.0430 |
| large_16 | 31.0707 | 17.5481 | 1.893× | 45.9893 | 37.8002 | 0.0863 | 0.1087 |
| long_short | 4.7096 | 2.6476 | 1.948× | 14.9678 | 8.0399 | 0.1362 | 0.1714 |
| tail_minus | 1.0701 | 0.4884 | 2.123× | 1.6118 | 0.6033 | 0.0089 | 0.0106 |
| tail_plus | 1.2613 | 0.4879 | 2.469× | 1.7208 | 0.6793 | 0.0094 | 0.0103 |

### metal

| workload | scalar total median ms | candidate total median ms | speed ratio | scalar p95 ms | candidate p95 ms | scalar enqueue median ms | candidate enqueue median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 0.4406 | 0.4505 | 0.941× | 0.6645 | 0.7120 | 0.1132 | 0.1223 |
| mixed_rank | 0.2193 | 0.1737 | 1.408× | 0.5730 | 0.4424 | 0.0149 | 0.0157 |
| large_1 | 0.5188 | 0.5046 | 1.020× | 0.8243 | 0.8404 | 0.0062 | 0.0069 |
| large_4 | 1.6065 | 1.6248 | 0.995× | 2.1504 | 2.2880 | 0.0267 | 0.0272 |
| large_16 | 5.4913 | 5.5111 | 0.994× | 6.2606 | 6.5923 | 0.0822 | 0.0938 |
| long_short | 1.0980 | 1.1555 | 0.963× | 1.8326 | 1.9717 | 0.1198 | 0.1430 |
| tail_minus | 0.4972 | 0.4749 | 1.035× | 0.8035 | 0.7865 | 0.0060 | 0.0067 |
| tail_plus | 0.4775 | 0.5007 | 0.954× | 0.7866 | 0.7836 | 0.0055 | 0.0064 |

### wgpu-msl

| workload | scalar total median ms | candidate total median ms | speed ratio | scalar p95 ms | candidate p95 ms | scalar enqueue median ms | candidate enqueue median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 0.6128 | 0.5783 | 0.978× | 0.7945 | 0.9427 | 0.1106 | 0.1227 |
| mixed_rank | 0.4257 | 0.4462 | 0.903× | 0.5076 | 1.3240 | 0.0137 | 0.0202 |
| large_1 | 0.6245 | 0.6229 | 1.011× | 0.8156 | 0.8373 | 0.0062 | 0.0066 |
| large_4 | 1.6276 | 1.5972 | 1.029× | 1.9262 | 2.1705 | 0.0200 | 0.0213 |
| large_16 | 5.6115 | 5.5067 | 1.014× | 6.8449 | 6.9049 | 0.0733 | 0.0747 |
| long_short | 1.1964 | 1.2391 | 0.959× | 1.5073 | 2.3492 | 0.1201 | 0.1319 |
| tail_minus | 0.6181 | 0.6202 | 0.997× | 0.8270 | 0.8223 | 0.0061 | 0.0065 |
| tail_plus | 0.6179 | 0.6101 | 1.018× | 0.8283 | 0.8233 | 0.0061 | 0.0065 |

## 当前候选的内存、首次执行和逐样本统计

这些独立进程同时记录Reference作背景；证明PR4独立收益仍使用上面的scalar COW对照。`allocs`为当前活跃分配数，peak为每池峰值之和。first_step包含编译、提交和执行，不是纯JIT。

### cpu

原始数据：[ pr4-production-cpu.json ](data/pr4-production-cpu.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 261.0191 | 263.0955 | 487.5564 | 0.0113 |
| tiny | fused | 0.1523 | 1.4549 | 6.2685 | 0.0106 |
| mixed_rank | reference | 0.0836 | 0.6515 | 14.9423 | 0.0114 |
| mixed_rank | fused | 0.0186 | 0.1834 | 12.7655 | 0.0115 |
| large_1 | reference | 0.0456 | 3.8885 | 14.2294 | 0.0110 |
| large_1 | fused | 0.0070 | 0.4620 | 0.5539 | 0.0112 |
| large_4 | reference | 13.1214 | 37.5781 | 56.1630 | 0.0130 |
| large_4 | fused | 0.0369 | 3.3783 | 13.9864 | 0.0168 |
| large_16 | reference | 161.8585 | 200.4224 | 250.7073 | 0.0120 |
| large_16 | fused | 0.1070 | 13.8374 | 26.1553 | 0.0120 |
| long_short | reference | 224.0489 | 255.9341 | 374.5010 | 0.0140 |
| long_short | fused | 0.1684 | 4.3006 | 19.3413 | 0.0109 |
| tail_minus | reference | 0.0552 | 15.3254 | 26.9394 | 0.0157 |
| tail_minus | fused | 0.0153 | 0.5490 | 0.8635 | 0.0065 |
| tail_plus | reference | 0.0524 | 19.7880 | 37.5428 | 0.0123 |
| tail_plus | fused | 0.0103 | 0.5401 | 0.7218 | 0.0107 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 120 | 1.875 | 8.000 | 266.247 |
| tiny | fused | 120 | 1.875 | 8.000 | 16.590 |
| mixed_rank | reference | 12 | 0.081 | 8.000 | 179.917 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 28.298 |
| large_1 | reference | 4 | 16.000 | 48.000 | 64.997 |
| large_1 | fused | 4 | 16.000 | 28.000 | 10.125 |
| large_4 | reference | 16 | 64.000 | 108.000 | 133.302 |
| large_4 | fused | 16 | 64.000 | 88.000 | 19.038 |
| large_16 | reference | 64 | 256.000 | 348.000 | 186.915 |
| large_16 | fused | 64 | 256.000 | 328.000 | 31.540 |
| long_short | reference | 128 | 33.875 | 68.000 | 263.523 |
| long_short | fused | 128 | 33.875 | 48.000 | 21.527 |
| tail_minus | reference | 4 | 16.000 | 48.000 | 77.907 |
| tail_minus | fused | 4 | 16.000 | 28.000 | 24.364 |
| tail_plus | reference | 4 | 16.000 | 56.000 | 71.055 |
| tail_plus | fused | 4 | 16.000 | 32.000 | 20.010 |

### metal

原始数据：[ pr4-production-metal.json ](data/pr4-production-metal.json)；features `std,cpu,metal,wgpu-msl,webgpu`。

| Workload | Mode | enqueue median ms | total median ms | total p95 ms | retained drop+sync median ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 1.0923 | 1.6277 | 3.3508 | 0.0450 |
| tiny | fused | 0.1206 | 0.4407 | 0.9038 | 0.0356 |
| mixed_rank | reference | 0.0991 | 0.5511 | 0.9191 | 0.0512 |
| mixed_rank | fused | 0.0138 | 0.1547 | 0.3580 | 0.0357 |
| large_1 | reference | 0.0294 | 1.5800 | 1.6896 | 0.0308 |
| large_1 | fused | 0.0061 | 0.5043 | 0.8151 | 0.0517 |
| large_4 | reference | 0.1425 | 5.8466 | 6.2149 | 0.0609 |
| large_4 | fused | 0.0378 | 1.6469 | 3.6393 | 0.0768 |
| large_16 | reference | 0.6604 | 23.3377 | 25.7558 | 0.0626 |
| large_16 | fused | 0.1003 | 5.7999 | 6.9125 | 0.0686 |
| long_short | reference | 1.0574 | 4.1014 | 6.2799 | 0.0580 |
| long_short | fused | 0.1286 | 1.1005 | 1.3556 | 0.0500 |
| tail_minus | reference | 0.0321 | 1.5923 | 2.0336 | 0.0462 |
| tail_minus | fused | 0.0269 | 0.5978 | 1.1394 | 0.0557 |
| tail_plus | reference | 0.0267 | 1.5662 | 2.0018 | 0.0451 |
| tail_plus | fused | 0.0063 | 0.4829 | 0.8020 | 0.0524 |

| Workload | Mode | live allocations | live MiB | pool peak upper bound MiB | first step ms |
|---|---|---:|---:|---:|---:|
| tiny | reference | 120 | 1.875 | 8.000 | 87.427 |
| tiny | fused | 120 | 1.875 | 8.000 | 7.486 |
| mixed_rank | reference | 12 | 0.081 | 8.000 | 168.361 |
| mixed_rank | fused | 12 | 0.081 | 8.000 | 10.029 |
| large_1 | reference | 4 | 16.000 | 38.000 | 77.846 |
| large_1 | fused | 4 | 16.000 | 28.000 | 7.266 |
| large_4 | reference | 16 | 64.000 | 98.000 | 98.641 |
| large_4 | fused | 16 | 64.000 | 88.000 | 8.127 |
| large_16 | reference | 64 | 256.000 | 338.000 | 71.764 |
| large_16 | fused | 64 | 256.000 | 328.000 | 14.477 |
| long_short | reference | 128 | 33.875 | 58.000 | 68.248 |
| long_short | fused | 128 | 33.875 | 53.000 | 38.053 |
| tail_minus | reference | 4 | 16.000 | 38.000 | 106.106 |
| tail_minus | fused | 4 | 16.000 | 28.000 | 6.822 |
| tail_plus | reference | 4 | 16.001 | 44.000 | 47.634 |
| tail_plus | fused | 4 | 16.001 | 32.000 | 1.861 |

## 首次执行、变体与代码体积的直接增量

使用与稳态相同的scalar/candidate二进制，每个workload/后端以独立进程重复3次并交替顺序。以下为首次完整step+sync时间的median；包括编译、执行与同步，不能解读为纯编译延迟。原始命令、输出和二进制体积见[pr4-first-step-and-size.json](data/pr4-first-step-and-size.json)。

| backend | workload | scalar first step ms | candidate first step ms |
|---|---|---:|---:|
| cpu | tiny | 56.488 | 80.171 |
| cpu | mixed_rank | 11.930 | 17.116 |
| cpu | large_1 | 11.836 | 10.343 |
| cpu | large_16 | 77.867 | 59.915 |
| cpu | tail_minus | 11.571 | 21.066 |
| cpu | tail_plus | 11.255 | 21.171 |
| metal | tiny | 9.231 | 6.505 |
| metal | mixed_rank | 11.565 | 11.581 |
| metal | large_1 | 6.852 | 6.317 |
| metal | large_16 | 16.717 | 14.730 |
| metal | tail_minus | 6.495 | 6.688 |
| metal | tail_plus | 6.962 | 6.330 |
| wgpu-msl | tiny | 6.778 | 6.645 |
| wgpu-msl | mixed_rank | 10.795 | 11.450 |
| wgpu-msl | large_1 | 7.209 | 6.742 |
| wgpu-msl | large_16 | 16.960 | 15.146 |
| wgpu-msl | tail_minus | 6.514 | 6.905 |
| wgpu-msl | tail_plus | 6.811 | 6.687 |

单独开启CubeCL编译/执行日志观察CPU tail_minus的三步运行，标量基线有2种AdamW编译签名（Initial/State），candidate有4种（Initial/State各自vector body和scalar tail）。主体实际宽度为16，尾部宽度1。该日志测量不用于性能计时，见[scalar](data/pr4-scalar-variant-proof.txt)和[candidate](data/pr4-candidate-variant-proof.txt)。跨全部AMSGrad/decay/layout/alias组合的可能变体更多，本数字只覆盖该实测workload。

- scalar：bench二进制 145,871,864 bytes；Mach-O __TEXT 97,337,344 bytes。
- candidate：bench二进制 145,815,896 bytes；Mach-O __TEXT 97,370,112 bytes。

体积是完整benchmark可执行文件（包含链接依赖和benchmark代码），不是device JIT kernel独立大小。两侧编译器剥离debug信息时都出现相同rust-objcopy/libLLVM警告，故体积比较只报告实际文件；不能外推发布包的精确增量。

小参数也纳入验收：CPU tiny/mixed/long_short的配对比值分别约1.950×/1.490×/1.948×。Metal tiny约0.941×、MSL mixed约0.903×，不是所有场景都更快；GPU实际仍width1，测得的小幅差异和首步变化完整列出，不把它们隐藏或归为CPU收益。
