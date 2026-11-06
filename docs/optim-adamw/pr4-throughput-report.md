# PR4：AdamW 大参数吞吐报告

状态：**CPU 向量化与 serial scalar tail 已实现；Metal/WGPU/MSL 保留 scalar variant。** 本报告只使用禁用 kache wrapper、绝对 beta cargo/rustc、独立 target 和匹配 source manifest 的 direct binaries。早期缓存构建数据全部标记 invalid，不参与结论。

## 结果

标量基线源码为 PR2 scalar COW；candidate 为当前 CPU-only vector body + serial scalar tail 实现。每个 workload 使用四轮交替 A/B、B/A、A/B、B/A，每侧每轮30样本。path proof 对8个 workload各输出8行 verified，并保存 binary/source hash。

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
- [旧 direct path proof](data/pr4-direct-final-path-proof.json)

早期 `pr4-verified-*`、`pr4-paired-*` 和 `pr4-direct-*` 的缓存构建结果均标记 `valid_for_acceptance=false`，因为 kache wrapper/target复用造成过binary/source归属冲突。它们仅作为历史排查记录。

## 实现

CPU runtime 根据所有输入/输出共同布局和 `io_optimized_vector_sizes` 选择宽度；不连续、偏移不对齐或不适合向量化时降到合法较小宽度。大连续 tensor 使用向量主体，剩余元素以 width=1 serial tail 处理，tail 传 runtime offset 和 units；kernel 在 offset 加法前排除 padded threads，避免接近 U32上限时回绕到已更新的开头。Metal/WGPU/MSL 强制 width=1。

COW planner未放宽，alias位置仍为param=0、m1=2、m2=3、max=4。主体和tail提交之间保持所有输入和输出生命周期，两个写区间不重叠。JIT回归验证vector和scalar variant key区分，runtime lr不改key。

实验/最终差异见 [pr4-vectorized.patch](experiments/pr4-vectorized.patch)，source hash见 [pr4-source-hashes.json](experiments/pr4-source-hashes.json)。当前 patch 可在 `target/adamw-pr2-source` scalar snapshot 中通过 `git apply --check`；当前工作树已包含该实现，不能直接重复应用。

## 验证

当前CPU/Metal/WGPU-MSL AdamW测试各36项通过，CPU JIT key测试单独通过；CPU odd lengths、offset、WGPU storage alignment、transpose/narrow、AMSGrad、cautious、overflow、COW、capture和checkpoint覆盖。CPU current direct test在禁用wrapper的独立target中通过。Vulkan缺SDK，CUDA/ROCm无硬件，浏览器WebGPU不可用。

已知边界：CPU结论只覆盖Apple M4 CubeCL CPU runtime，GPU scalar结论只覆盖本机Metal/WGPU-MSL。约16GiB U32极限buffer仅做源码算术核验，未实机分配。

PR4 的新版本仍需完成两轮独立 code review；性能门槛已由 direct source-matched数据满足CPU目标，GPU保守回退保持正确性。
