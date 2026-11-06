# PR3 独立下游：无根 `[patch]` 的本地 git 验证

本目录验证包含 getter 的 CubeCL、同步依赖的 CubeK、应用 PR3 候选的 Burn 可以通过普通
`git + rev` 依赖供独立消费者使用。**这是本地 git 依赖结构验证，不证明远程可获取 revision，
不代表 PR3 完成。** 远程 getter 发布及 CubeK/Burn 生产依赖接入仍未授权；本任务没有发布 remote。

## 复现

在当前 Burn 仓库根目录执行，要求 Python 3.9+、已安装的 beta Rust 工具链、现有 git 用户身份，
以及 Cargo git cache 中的两个锁定 revision。脚本不依赖 `tomllib` 或第三方 Python 包，
所有 subprocess 命令均使用参数列表，不经 shell。

```sh
python3 docs/optim-adamw/prerequisites/downstream/verify.py \
  --run-dir target/adamw-downstream-reproduce --jobs 2
```

目标目录必须不存在，且必须是当前仓库 `target/adamw-downstream-*` 的直接子目录。
省略 `--run-dir` 时使用 UTC 时间生成目录名。可用 `--metadata-only` 只验证依赖图；默认先保存并
报告 metadata 结果，再继续 CPU 编译和运行。失败保留命令退出码、完整 stdout/stderr 及 traceback，
不会将编译或运行失败记成通过。每次执行更新本目录的 `validation.json` 和 `SHA256SUMS`，
各次运行的独立副本保存在自己的 target 目录；逐条命令详情及退出码在该目录的
`commands.json`，完整输出在 `logs/`。用 `--previous-run target/adamw-downstream-<旧目录>`
可在新摘要中引用前次尝试及 hash，不覆盖旧证据。

脚本执行如下步骤：

1. 只读查找已缓存的 CubeCL `d88b0febb50e7b83a8548125f6a701c12f11a2ba` 与
   CubeK `3d3ab9680931c5903286bb30eb2b8ff156e1618b`，用 `git archive` 展开干净副本。
2. CubeCL 应用上级目录的 `cubecl-execution-stream.patch`，创建没有 remote 的本地 git commit。
   CubeK workspace 的 `cubecl`、`cubecl-common`、`cubecl-environment`、`cubecl-ir`
   全部改为该 CubeCL 的同一 `file://` URL 与完整 commit revision，然后创建本地 commit。
3. 从本仓库 HEAD 的 `git archive` 创建 Burn 副本，依次应用
   `experiments/pr3-local/foreach.patch` 与 `refinement/candidate-refinement.patch`。
   Burn workspace 的四项 CubeCL 依赖（包括 `cubecl-zspace`）及 CubeK 依赖统一到对应本地 URL/rev。
   主工作区未提交改动不进入副本。
4. 创建自带 `[workspace]` 的消费者，只有一项直接依赖：上述本地 revision 的 `burn` facade，
   `default-features = false`，启用 `std,cpu,optim,cubecl`。消费者没有 path 依赖或 `[patch]`。
   Cargo 正常 fetch 本地 git、解析独立 lock；编译使用 `--locked --offline`。
5. `cargo metadata --format-version 1` 检查图中每个 CubeCL/CubeK/Burn 包只有一个预期 source，
   并检查 Burn 与 CubeK 实际依赖边共享同一 CubeCL package ID。`cubecl-hip-sys` 必须来自 registry。
   未进入消费者依赖图的 CubeCL workspace 包单独记录，不将其记为编译或依赖覆盖。
6. 通过公开 `burn::module`、`burn::optim`、`burn::tensor`、`burn::cubecl::logging` API
   运行两个 `[2,2]` F32 参数的 Module。先以 Fused 单参数 step 校准 observer（batch=0、single=1），
   然后要求 Foreach 两步各 batch=1、single=0。两步均核对 Reference 结果和独立 f64 标量公式，
   Module 参数位于 CPU autodiff device，梯度位于其 inner CPU device；同时检查 shape、dtype、
   device、参数 ID、require-grad。使用 AMSGrad、非零 weight decay；第二步梯度反向减半，
   会区分遗漏旧 momentum state 等错误。所有失败均为断言，没有能力不足时的提前返回。

## 环境与写入边界

使用 `rustup which --toolchain beta` 得到绝对 cargo/rustc/rustdoc/rustfmt 路径。
`RUSTC_WRAPPER`、`RUSTC_WORKSPACE_WRAPPER` 及对应 Cargo build wrapper 环境变量为空，
移除继承的 Cargo runner 环境变量；最终直接启动构建产物，记录真实退出码。
检查 Cargo 会发现的祖先及用户配置，遇到 source/patch/replace/paths 覆盖或配置 runner 即拒绝执行。
不传任何 Cargo `--config` 参数。既有用户配置的 Rust wrapper 被空环境变量覆盖，
HOST_CC/HOST_CXX 显式使用 `cc`/`c++`。

快照、tar、消费者源码/lock、编译产物、临时文件、CubeCL environment 和完整日志都在本次
`target/adamw-downstream-*` 中。允许 Cargo 正常 fetch/cache，不修改其依赖源码。
已有 LLVM bundler 安装作为构建前置读取；未复制或修改其源码。脚本没有修改生产源码或根
Cargo.toml/Cargo.lock 的步骤，并记录这两个文件前后 hash。未运行全仓库 tests 或额外硬件后端测试。

本地 git 使用现有用户的 name/email，不签名、不执行 hooks，不增加 AI commit trailers。
commit 时间固定为 Burn 基准 HEAD 的提交时间；URL 包含本次目录，换目录会改变 CubeK/Burn 的
manifest、tree 和 commit hash，这是预期行为。每个输入 archive、最终 archive、补丁、manifest、
工具版本、命令参数和真实 local git SHA 均在验证摘要中记录。

## 结果与证据

**最终 metadata、CPU build、实际 Foreach runtime 均通过（退出码 0）。**
当前运行目录：`target/adamw-downstream-localgit-20261106-v2/`。
完整结果见 [validation.json](validation.json)，文件校验清单见 [SHA256SUMS](SHA256SUMS)。
本次脚本、消费者及报告由 Codex 编写和执行，未创建或更新远程 PR。

| 实际检查 | 结果 |
| --- | --- |
| Python 3.9 语法解析 | 通过；脚本由本机 Python 3.9 执行 |
| getter、foreach、refinement 补丁 `git apply --check` | 全部通过 |
| 独立消费者无 patch/config 覆盖的 metadata graph | 通过 |
| beta `cargo build --locked --offline` | 通过，约 92 秒，独立 target、jobs=2 |
| Fused 正向校准 | batch=0、single=1、total=1 |
| Foreach 第一步 / 第二步 | 各 batch=1、single=0、total=1 |
| 两参数 × 两步的数值与 metadata | Reference 与独立标量公式均匹配；最大绝对误差 `2.564828776030481e-7` |
| 根 Cargo.toml / Cargo.lock 前后 hash | 均相同 |

工具链为 `cargo/rustc/rustdoc 1.100.0-beta.7`，主机为 aarch64 macOS。
未执行全仓库检查、GPU 后端、性能测试或主线程负责的 Remote 测试，均不在本次通过声明中。

本次基准 Burn HEAD 为 `4cc00ce82099de74c48c78a44f625960b2d16464`。URL 均以
`file:///Volumes/HIKSEMI/code/burn/target/adamw-downstream-localgit-20261106-v2/` 开头：

| 快照（URL 后缀） | 本地 git revision |
| --- | --- |
| `cubecl` | `c3dd9176fbc7f476b5f28fab6fbdf6540bd0dac2` |
| `cubek` | `aafe7c1e99f29d32a8c7c2c5e1fe65b5c03e236c` |
| `burn` | `34c8c571f9cc5e969d7153bf327369ca15e9329a` |

metadata 共 653 个包，22 个 CubeCL 包各只有预期的单一 source。
Burn/CubeK 分别有 14/25 条直接依赖 CubeCL 的边，交集为同一 package ID 的
`cubecl`、`cubecl-common`、`cubecl-environment`。`cubecl-hip-sys` 来自 registry，
`cubecl-inspect` 没有进入消费者依赖图。所有 source、依赖边及 features 的完整数据保存在
`logs/metadata.json` 与 `logs/graph-summary.json`，摘要包含相应 hash。

首次尝试 `target/adamw-downstream-localgit-20261106/` 的 metadata 和 build 成功，但 runtime
退出 **101**：Fused observer 校准已经观察到 single=1，随后普通 CPU 参数进入 ModuleOptimizer
触发 `burn-dispatch/src/backend.rs:783` 的 `Requires autodiff tensor`。对照候选的 Module 测试及
`ParameterContext::capture` 调用，修正消费者为 autodiff Module 参数和 inner CPU 梯度。
没有修改候选实现。原消费者、binary 和 `logs/055-adamw-downstream-check.*.log` 均保留，
原始脚本与完整摘要均实际归档在**首次运行的原 target 目录**，相对链接为
[01-verify.py](../../../../target/adamw-downstream-localgit-20261106/attempts/01-verify.py) 和
[01-validation-full.json](../../../../target/adamw-downstream-localgit-20261106/attempts/01-validation-full.json)。
首次失败没有计为运行通过。

最终摘要的 `cargo_configs` 只保留配置文件路径、SHA-256 和无 source override/runner 的审计结果，
不收录个人配置正文。`script_sha256` 对应实际执行时归档的 `executed_script_path`；
`report_script_sha256` 对应仅改进配置脱敏和归档的当前脚本。CPU 测试结果仍对应原执行脚本及
同一 consumer/binary hash，没有将报告整理称为重新运行。

从仓库根目录核验归档：

```sh
shasum -a 256 -c docs/optim-adamw/prerequisites/downstream/SHA256SUMS
```

这里只验证 CPU 上最小真实 Foreach 的依赖、数值和 launch 路径；不是性能测量、完整 PR3 功能矩阵、
Remote 回退测试或最终 review 门禁。主线程的 Remote 验证与本任务独立。
