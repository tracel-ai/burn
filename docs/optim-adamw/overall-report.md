# AdamW 优化总报告

状态：PR1、PR2 已完成并通过各自连续两轮 review；PR3 按设计门禁停止；PR4 CPU 向量化与serial scalar tail已完成直接source-matched A/B，GPU runtime保留scalar。CPU large参数与tail均有收益；PR4最新实现已连续两轮review通过；PR3尚未实现，全部实现总目标尚未完成。

## 交付状态

| 阶段 | 结果 | 证据 |
|---|---|---|
| PR1 单参数 Fused | 已完成 | [PR1报告](pr1-fused-report.md)，[审查记录](review-log.md)；两轮最终 PASS |
| PR2 COW 输出复用 | 已完成 | [PR2报告](pr2-cow-report.md)；两轮最终 PASS |
| PR3 Foreach | 停止/阻塞 | [PR3停止报告](pr3-foreach-report.md)；两轮确认当前 CubeCL缺少合格 stream getter |
| PR4 大参数吞吐 | CPU实现及两轮review完成 | [PR4报告](pr4-throughput-report.md)；CPU大参数及serial tail均有收益，GPU保留scalar |
| 分支最终三轮 review | 准备开始，0/3 | 按当前分支完整审查 |

PR3 的阻塞来自锁定 CubeCL revision：`Client::stream_id()` 私有，`service_stream()` 和 `ServiceStream` 为 `pub(crate)`。公开的 profile、memory report、empty 和 stream-mode API 都会阻塞、分配或改变运行模式；`Handle.stream` 只是 allocation 创建 stream。最小前置是上游增加无副作用 `Client::execution_stream()` getter，随后才可实现安全 Foreach。

PR4实现共同vector width、serial scalar tail、布局/offset fallback、JIT key与U32 tail防回绕。禁用kache并独立target重建后，CPU large_1/4/16约2.63×/2.16×/1.89×，tail约2.12×/2.47×；Metal/MSL没有稳定收益，当前GPU强制scalar。旧缓存停止结论无效；最终CPU版本已连续两轮review通过。

## 当前生产实现

PR1 为 F32 raw CubeCL 单参数 fused AdamW：moment、AMSGrad、cautious decay、bias correction 和参数更新在一次 kernel 中完成；Fusion、autodiff direct step、非 F32、空/rank0、tiled/quantized、不匹配 service/span/binding 等情况在 launch 前回退。

PR2 为每个输出独立的 COW planner：只在 `can_mut + non-overlapping + contiguous + 独立 service/allocation` 时复用 param/m1/m2/max；wrapper 转换完成后释放临时引用；primitive 拒绝则重建完整 fallback state。capture/replay、alias、共享输入、AMSGrad、checkpoint 和布局测试覆盖。

## 主要性能证据

PR1 在 Apple M4 上对 raw CubeCL CPU、Metal、WGPU/MSL、WGPU/WGSL 均有稳定结构收益，详情见 PR1 原始 JSON 和报告。PR2 在 large_16 上降低池峰值上界：CPU 448→328 MiB，Metal 343→328 MiB；CPU 独占稳态时间多数 workload 改善，GPU 基本持平。共享 CPU tiny 有约 7% 中位时间增加，报告如实保留该限制。

PR4早期paired及verified数据均标记`valid_for_acceptance=false`，原因是cached binary/source hash冲突。当前只采用绝对toolchain cargo/rustc、禁用wrapper、独立target的serial-tail direct数据，path proof绑定binary与source hash。PR4最终两轮review已通过。

## 验证

已通过或完成：

- 当前CPU、Metal、WGPU/MSL、WGPU/WGSL各36项测试通过（1项进程级JIT测试默认忽略，CPU单独串行执行通过）。这些测试使用禁用wrapper的直接toolchain构建。
- 当前CPU Fusion直接构建36项通过；CPU raw all-targets clippy直接构建通过，原始摘要见[pr4-current-validation.json](data/pr4-current-validation.json)。
- PR1/PR2 的四 runtime `--verify-path`、内存、稳态、JIT 和 allocation ID 原始数据。
- PR3 pinned CubeCL API静态核查两轮；PR4当前CPU-only serial-tail实现及完整测量报告已完成Darwin、Sagan连续两轮PASS。

已知未通过或未可用：

- `cargo run-checks` 的格式、typos、cargo audit 已通过；Workspace lint 失败于未改动的 `burn-train/src/renderer/tui/recent_history.rs:206`（needless_range_loop）及 `burn-store/src/filter.rs:178`（needless_bool），报错文件相对HEAD无diff。unused dependency仅是附带警告。完整日志见 [final-run-checks.txt](data/final-run-checks.txt)；后续仓库级feature与backend测试未运行。
- `cargo check -p burn-optim --no-default-features` 仍被未改动的 ModuleOptimizer 对 std gated `lr_scheduler` 的无条件引用和 alloc 导入问题阻断，PR1报告中已有记录。
- Vulkan 缺少 macOS Vulkan SDK；CUDA/ROCm 无本机硬件；浏览器 WebGPU API 不可用；Flex 没有 CubeCL Fused路径。

最终三轮分支 review 将审查当前生产状态、所有停止报告、性能证据和文档一致性；每轮若发现问题会先修复再重新计数。

## 当前累计效果（直接构建）

以下仅使用当前候选同一二进制里的Reference/Fused比较，包含Fused+COW+CPU vector的累计结果，
不能把累计收益归给某一个PR。每workload预热5步、采30样本并逐步sync；CPU存在系统调度噪声，
尤其小参数比值不应外推到其他机器。数据见[CPU](data/pr4-production-cpu.json)、[Metal](data/pr4-production-metal.json)。

| backend | workload | Reference median ms | final Fused median ms | ratio |
|---|---|---:|---:|---:|
| cpu | tiny | 263.0955 | 1.4549 | 180.84× |
| cpu | mixed_rank | 0.6515 | 0.1834 | 3.55× |
| cpu | large_1 | 3.8885 | 0.4620 | 8.42× |
| cpu | large_4 | 37.5781 | 3.3783 | 11.12× |
| cpu | large_16 | 200.4224 | 13.8374 | 14.48× |
| cpu | long_short | 255.9341 | 4.3006 | 59.51× |
| metal | tiny | 1.6277 | 0.4407 | 3.69× |
| metal | mixed_rank | 0.5511 | 0.1547 | 3.56× |
| metal | large_1 | 1.5800 | 0.5043 | 3.13× |
| metal | large_4 | 5.8466 | 1.6469 | 3.55× |
| metal | large_16 | 23.3377 | 5.7999 | 4.02× |
| metal | long_short | 4.1014 | 1.1005 | 3.73× |

实现、测试、测量脚本、报告和review由编码agent完成，没有声称人工review。用户自行创建并推送的
`ba84da059`保存了此前状态；后续每个通过门禁的阶段将按明确授权提交并推送到同一fork分支。
PR3仍未实现，因此本报告不能作为“全部PR优化已完成”的证明。
