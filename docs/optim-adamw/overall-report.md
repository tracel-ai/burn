# AdamW 优化总报告

状态：PR1、PR2 已完成并通过各自连续两轮 review；PR3 按设计门禁停止；PR4 CPU 向量化与serial scalar tail已完成直接source-matched A/B，GPU runtime保留scalar。CPU large参数与tail均有收益；PR4最新实现仍待连续两轮review，全部实现总目标尚未完成。

## 交付状态

| 阶段 | 结果 | 证据 |
|---|---|---|
| PR1 单参数 Fused | 已完成 | [PR1报告](pr1-fused-report.md)，[审查记录](review-log.md)；两轮最终 PASS |
| PR2 COW 输出复用 | 已完成 | [PR2报告](pr2-cow-report.md)；两轮最终 PASS |
| PR3 Foreach | 停止/阻塞 | [PR3停止报告](pr3-foreach-report.md)；两轮确认当前 CubeCL缺少合格 stream getter |
| PR4 大参数吞吐 | CPU候选待验收 | [PR4报告](pr4-throughput-report.md)；CPU大参数有收益，尾部退化仍待解决，GPU保留scalar |
| 分支最终三轮 review | 尚未开始 | 完成 PR4停止审查后执行 |

PR3 的阻塞来自锁定 CubeCL revision：`Client::stream_id()` 私有，`service_stream()` 和 `ServiceStream` 为 `pub(crate)`。公开的 profile、memory report、empty 和 stream-mode API 都会阻塞、分配或改变运行模式；`Handle.stream` 只是 allocation 创建 stream。最小前置是上游增加无副作用 `Client::execution_stream()` getter，随后才可实现安全 Foreach。

PR4实现共同vector width、serial scalar tail、布局/offset fallback、JIT key与U32 tail防回绕。禁用kache并独立target重建后，CPU large_1/4/16约2.63×/2.16×/1.89×，tail约2.12×/2.47×；Metal/MSL没有稳定收益，当前GPU强制scalar。旧缓存停止结论无效；新CPU候选仍待两轮review。

## 当前生产实现

PR1 为 F32 raw CubeCL 单参数 fused AdamW：moment、AMSGrad、cautious decay、bias correction 和参数更新在一次 kernel 中完成；Fusion、autodiff direct step、非 F32、空/rank0、tiled/quantized、不匹配 service/span/binding 等情况在 launch 前回退。

PR2 为每个输出独立的 COW planner：只在 `can_mut + non-overlapping + contiguous + 独立 service/allocation` 时复用 param/m1/m2/max；wrapper 转换完成后释放临时引用；primitive 拒绝则重建完整 fallback state。capture/replay、alias、共享输入、AMSGrad、checkpoint 和布局测试覆盖。

## 主要性能证据

PR1 在 Apple M4 上对 raw CubeCL CPU、Metal、WGPU/MSL、WGPU/WGSL 均有稳定结构收益，详情见 PR1 原始 JSON 和报告。PR2 在 large_16 上降低池峰值上界：CPU 448→328 MiB，Metal 343→328 MiB；CPU 独占稳态时间多数 workload 改善，GPU 基本持平。共享 CPU tiny 有约 7% 中位时间增加，报告如实保留该限制。

PR4早期paired及verified数据均标记`valid_for_acceptance=false`，原因是cached binary/source hash冲突。当前只采用绝对toolchain cargo/rustc、禁用wrapper、独立target的serial-tail direct数据，path proof绑定binary与source hash。尚未完成最终两轮review。

## 验证

已通过或完成：

- 当前生产 raw CPU AdamW 35 项测试；CPU Fusion 35 项测试；CPU raw all-targets clippy；格式检查。
- 当前生产 Metal 34 项及 capture/replay 测试；WGPU/MSL 与 WGPU/WGSL 各 35 项测试；独立 vector/JIT 实验测试在撤回前通过。
- PR1/PR2 的四 runtime `--verify-path`、内存、稳态、JIT 和 allocation ID 原始数据。
- PR3 pinned CubeCL API 静态核查两轮；PR4 实验数据按独立target复现，停止决定正在审查。

已知未通过或未可用：

- `cargo run-checks` 的格式、typos、cargo audit 已通过；Workspace lint 失败于未改动的 `burn-train/src/renderer/tui/recent_history.rs:206`（needless_range_loop）及 `burn-store/src/filter.rs:178`（needless_bool），报错文件相对HEAD无diff。unused dependency仅是附带警告。完整日志见 [final-run-checks.txt](data/final-run-checks.txt)；后续仓库级feature与backend测试未运行。
- `cargo check -p burn-optim --no-default-features` 仍被未改动的 ModuleOptimizer 对 std gated `lr_scheduler` 的无条件引用和 alloc 导入问题阻断，PR1报告中已有记录。
- Vulkan 缺少 macOS Vulkan SDK；CUDA/ROCm 无本机硬件；浏览器 WebGPU API 不可用；Flex 没有 CubeCL Fused路径。

最终三轮分支 review 将审查当前生产状态、所有停止报告、性能证据和文档一致性；每轮若发现问题会先修复再重新计数。
