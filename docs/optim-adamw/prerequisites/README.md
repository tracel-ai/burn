# PR3 前置：CubeCL execution stream getter

这是解除 PR3 API 门禁的可复现最小补丁，不是 Foreach 实现，也没有改变 Burn 的生产依赖。
补丁基于锁定的 CubeCL `d88b0febb50e7b83a8548125f6a701c12f11a2ba`，文件见
[cubecl-execution-stream.patch](cubecl-execution-stream.patch)。

## 接口与验证

新增 `Client::execution_stream()`，直接调用内部现有 `stream_id()` resolver。
它不向 server 提交工作、不分配设备内存、不改变 pin 或 scope；未 pinned 的 client 每次查询
解析当前 scope，pinned client 始终返回固定 stream。resolver 原有的 TLS/原子初始化行为保留。

CPU integration tests 覆盖：

- 未 pinned 的 client 和 clone 在两层 scope 中跟随当前 stream，退出后恢复原 scope。
- pinned client 和 clone 在不同 scope 中始终返回 pin，原未 pinned client 不受影响。
- getter 返回值与随后实际创建的 allocation 的 stream 一致；这是验证结果的独立观测，
  不是把 allocation 创建 stream 当作生产查询的替代接口。

使用绝对 beta cargo/rustc、禁用 rustc wrapper、取消测试 runner、独立 target 构建，两个测试均通过。
源码、补丁与日志哈希见 [validation.json](validation.json)，测试结果见 [test-result.txt](test-result.txt)。
该证据只覆盖新增 getter 的 CPU行为；不是 Foreach 的数值、分组、stream恢复或性能验收。
两个变更文件的rustfmt检查和补丁对干净pin的apply检查通过。clippy未通过：
原有`cubecl-cpu/src/compute/threadpool/scheduler/dispatcher.rs:69`触发`needless_range_loop`；
该文件与锁定revision逐字一致，错误及哈希已记录，未当作clippy通过。

## 复现

把 CubeCL 的指定 revision 展开到当前 Burn 仓库的 `target/` 子目录，对该副本应用补丁。
不要修改 Cargo 缓存中的依赖源码。示例中的 `<beta>` 是实际 Rust beta toolchain 目录。

```sh
# 在 Burn 仓库根目录执行；给 git apply 显式指定副本路径。
git apply --directory=target/adamw-cubecl-stream-prerequisite \
  docs/optim-adamw/prerequisites/cubecl-execution-stream.patch
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= \
  RUSTC=<beta>/bin/rustc RUSTDOC=<beta>/bin/rustdoc \
  CARGO_TARGET_DIR=target/adamw-stream-prerequisite-build \
  <beta>/bin/cargo test --manifest-path target/adamw-cubecl-stream-prerequisite/Cargo.toml \
    -p cubecl-cpu --no-default-features \
    --features std --test execution_stream
```

测试用副本已经应用补丁时不要重复应用；归档补丁可对干净的指定 revision 使用。

## 仍需满足的依赖前置

此次重新查询的官方 CubeCL head 是 `565108e077557f634aa7f5bf19b34ed655600ba7`，
比当前 pin 新16个提交，但 getter 仍未公开。因此单纯升级到该 head 不能解除门禁。

Burn workspace 的四项 CubeCL dependency 与 pinned CubeK workspace 的四项 CubeCL dependency
都引用同一旧 revision。只改 Burn 的 pin 会引入两套 CubeCL 类型，不能作为完成升级的证据。
需要包含 getter 的可获取 CubeCL revision，并同步 CubeK 与 Burn 的依赖关系，再验证依赖图只有
预期的一套类型。该升级不包含在本补丁中。

临时 root `[patch]` 或修改本机缓存只能让当前 checkout 编译，不能证明 Burn 作为下游依赖时可用。
本仓库未发布其他仓库的分支或修改上游 pin；PR3 当前仍受此前置阻塞。前置可用后才继续完整的
Sequence原型、模块两次遍历、分组/chunk、fallback、checkpoint、两轮review和性能对照。

本补丁及验证材料经James、Ptolemy两轮独立agent review通过。审查范围仅为getter前置，
不是Foreach或总体目标验收；记录见[review-log.md](../review-log.md)。
