# Foreach decorator 验证包

本目录是独立 Cargo workspace，只通过公开 API 验证 PR3 本地候选的 Group、Capture 与可选 Remote
回退。`Cargo.toml` 的 `[workspace]` 隔离主仓库；保留本目录的 `Cargo.lock`，以
`cargo test --locked` 固定独立依赖解析。没有新增生产 feature 或测试入口，没有修改主仓库
`Cargo.lock`、生产代码、bench 或 Cargo 缓存源码。测试与说明由 Codex 实现，不计最终 review。

## 复现

在 [base patch](../foreach.patch) 对应的 Burn 基线 `3221d3f9d` 上，从仓库根目录执行。
已经应用候选补丁并完成本地依赖准备的工作树直接执行最后一条命令，避免重复应用。

```sh
git apply --check docs/optim-adamw/experiments/pr3-local/foreach.patch
git apply docs/optim-adamw/experiments/pr3-local/foreach.patch
git apply --check docs/optim-adamw/experiments/pr3-local/refinement/candidate-refinement.patch
git apply docs/optim-adamw/experiments/pr3-local/refinement/candidate-refinement.patch
python3 docs/optim-adamw/experiments/pr3-local/prepare.py /path/to/existing-cubecl-checkout
bash docs/optim-adamw/experiments/pr3-local/checks/run.sh
```

[prepare.py](../prepare.py) 需要包含锁定 revision 的现有 CubeCL git checkout，创建
`target/adamw-foreach-cubecl` 与 `target/adamw-foreach-local.toml`，并拒绝覆盖已有目标。
不要应用用于性能分项的 `phase-profile.patch`。本 harness 不依赖该临时插桩。

`run.sh` 默认用 `rustup which --toolchain beta` 分别解析 cargo、rustc、rustdoc 的绝对路径。
可通过 `ADAMW_CHECKS_TOOLCHAIN=<已安装的工具链>` 覆盖；脚本不包含用户目录硬编码。
脚本清空两个 Rust wrapper，固定 `BURN_DEVICE=cpu`，使用独立
`target/adamw-foreach-decorators`，并将仓库的本地 override 以**绝对 `--config` 路径**传入。
全部用例串行运行，没有 ignore、过滤或硬件 early-return。profile 关闭 strip，避免本机 beta
`rust-objcopy` 缺失 `libLLVM.dylib` 的 strip 警告。

启用本机loopback WebSocket Remote验证（服务只监听`127.0.0.1`随机端口，不联系外部服务）：

```sh
ADAMW_CHECKS_FEATURES=remote ADAMW_CHECKS_LOG_NAME=remote \
  bash docs/optim-adamw/experiments/pr3-local/checks/run.sh
```

默认运行Group/Capture两项；`remote` feature追加第三项，并带入仅此harness使用的可选tokio依赖。
脚本构造始终非空的参数数组，兼容macOS Bash 3.2的`set -u`；features作为一个参数传递。

## 测试范围

- **Group 实际数值执行**：两个成员均为本机 CPU device 的真实 Group；模块含两个 `[2,2]`
  F32 参数，AMSGrad 开启，执行两步。Reference 与 Foreach 比较参数并检查 require-grad。
  初始 tracked device 的 checkpointing strategy 必须存在且不是 `Disabled`；两种执行模式的
  每个参数在每步结束后分别与该初始策略比较，防止共享恢复路径同时丢失策略仍然通过。
  direct `AdamWState` 检查全部 m1/m2/max 的值、time 和 Group device；
  模块通过公开 burnpack Reader 检查六个 state tensor 的值、ParamId、名称、rank、path 与
  time。第二步使用反向、减半梯度，并检查手算 moments；两步之间重载 Foreach 模块 state。
  模块内部 state 的 device 没有公开 getter，因此 device 的直接断言来自 direct state，模块
  另验证恢复后的更新行为及参数 device。此用例不证明不同物理设备或分片布局的行为。
- **Capture 实际 IR 录制**：使用 `Device::capture()`，对两个参数各做两步 direct AdamW。
  声明最终参数、m1、m2、max 共八个图输出，检查 time、shape、dtype 和 Capture device。
  仅规范化全局分配的 tensor ID，保留其余 operation、scalar、shape、status、输出顺序；比较
  Reference/Foreach 的完整 IR 与初始化值。Capture 不执行数值计算，此结果不能声称 Capture
  数值运行通过，也不是 GPU hardware graph capture/replay 的证据。
- 两个用例都先用 raw CPU direct Foreach 更新作 observer 正向对照，必须看到一个 single
  AdamW launch。Group 允许 Reference 张量算子的实际 launch，但 AdamW batch/single 必须为
  零；Capture 的所有 raw kernel launch 必须为零。

- **Remote 实际数值执行**：启动本机WebSocket server，实际服务CPU后端，明确断言远端dispatch。
  比较远端Reference/Foreach模块、远端Foreach direct与本地Reference的两步更新。
  检查参数、六个state值、time、rank、path、ParamId、dtype，direct输出/state保留Remote device，
  module输出保留原autodiff与非Disabled的初始checkpointing策略；两步间重新加载checkpoint。
  moments另用手算值核对，不能仅凭两种模式相同判定正确。专用batch/single launch为零。
  整个Remote用例包含30秒watchdog，初始化/read/sync卡住时测试进程失败退出；不能将server的Drop
  当作打断阻塞读的手段。仅覆盖同机CPU服务端，不外推真实跨机网络、其他传输或服务端GPU。

特殊值、rank 8 checkpoint 迁移、预算边界和性能分项由主线程负责，见refinement报告。

## 实际结果与证据

2026-11-06 14:54:44 UTC，Darwin arm64，cargo/rustc/rustdoc `1.100.0-beta.7`：
补充 checkpointing 独立断言后重跑，**2 passed，0 failed，0 ignored，0 filtered**，
测试执行 0.23 秒。

新增Remote后的最终运行：默认入口 **2 passed，0 failed，0 ignored，0 filtered（0.26秒）**；
启用Remote并加入watchdog后 **3 passed，0 failed，0 ignored，0 filtered（0.27秒）**。
早期默认入口曾因空array与Bash 3.2 nounset不兼容退出1，修复后实际复测通过，保留失败摘要。

| 检查 | 观测 |
|---|---|
| Group | 两步 time 为 1、2；参数及六个 state tensor 检查通过；Reference/Foreach 各自保留初始 `Balanced` checkpointing 策略；448 次 Reference 算子 launch；AdamW batch=0、single=0 |
| Capture | 62 个完全一致的规范化 IR operation，8 个输出，time=`[[1,1],[2,2]]`；所有 raw launch=0 |
| Remote | 两参数两步、远端/本地数值与六个state一致，time为1/2，context为Balanced；batch=0、single=0；含本地参考的raw算子总计224 |
| observer 正向对照 | 每个用例开始前 batch=0、single=1 |
| 退出状态 | 本次默认/Remote真实用例脚本均=0；新增feature参数后重跑无效target探针，Cargo=101、tee=0、脚本=101 |
| 静态检查 | `bash -n run.sh` 与 beta `rustfmt --check` 通过 |
| 主仓库 lock | 本次重跑前后 SHA-256 一致 |

- [decorators.log](logs/decorators.log)：此前Group/Capture两项运行的工具链、实际输出、退出状态与原始日志 hash。
- [remote.log](logs/remote.log)：新增Remote及watchdog后的三项最终实跑结果。
- [default-after-remote.log](logs/default-after-remote.log)：保持默认feature时两项实际通过。
- [default-empty-args-failure.log](logs/default-empty-args-failure.log)：已修复的Bash空array失败证据。
- [remote-exit-status-probe.log](logs/remote-exit-status-probe.log)：新增feature参数后再次验证错误退出101。
- [remote-validation.json](remote-validation.json)：当前Remote增量的源码、日志hash、原始失败与修复记录；生产源码hash保持候选版本。
- [capture-reference-ir.json](logs/capture-reference-ir.json) 与
  [capture-foreach-ir.json](logs/capture-foreach-ir.json)：完整规范化 IR、初始化值、输出名称和 time。
- [exit-status-probe.log](logs/exit-status-probe.log)：故意失败的 Cargo 调用，专门证明状态传播，
  不计为 decorator 用例失败。
- [validation.log](logs/validation.log)：脚本/格式检查及主仓库 lock hash。
- [build-history.log](logs/build-history.log)：首轮 harness API 编译错误及修正后首次通过的简短记录、
  原始日志 hash，以及 checkpointing 独立断言修正前的证据局限与日志 hash。
  首轮误用私有 `Shape.dims`，已改为公开 `dims()`。
- [SHA256SUMS](SHA256SUMS)：冻结的 harness 与可审阅证据文件 hash。

`run.sh` 将完整 Cargo 输出保存在本地 `logs/*.raw.log`；这些 noisy 日志由本目录
`.gitignore` 排除，只保留摘要及 SHA-256 作为审阅材料。重新运行会更新结果文件；本次冻结的
证据对应 `SHA256SUMS`。依赖本地 path 的复现仍要求上述 Burn 补丁与 CubeCL override 一致。
