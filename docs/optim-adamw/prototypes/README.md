# PR3 Sequence 参数形式验证

此目录保留设计第8.2节要求的本地原型。它在CPU上实际编译并执行，验证
`Sequence<LinearView>`、`ComptimeOption<Sequence<LinearView>>` 和
`Sequence<LinearViewMut>` 可组合用于一次多张量chunk launch。
它不是AdamW内核、模块级Foreach实现或性能优化交付。

## 覆盖与结果

- 三个独立allocation，shape分别为`[21]`、`[3,5]`、`[1,2,2,3]`。
- `[3,5]`输入stride为`[1,3]`，输出连续；主机独立计算逻辑索引对应的物理位置作为期望值。
- 每chunk最多8个元素，七个chunk对应七个cube，一次launch覆盖全部张量。
  local offset显式加入输入与输出index，最后的不足8元素chunk有边界检查。
- 公式仅为`out = input * 2 + optional_history`，易于逐值精确检查。
- history包含张量编号：`tensor * 1000 + i + 1`。不同张量即使逻辑位置相同，历史值也不同，
  因而误读另一张量的history会导致数值检查失败。
- 有/无history、out-of-place/部分alias共四种配置，每种以新allocation重复两次；
  observer断言每次实际仅一个kernel，kernel key集合确认四种variant分开、重复配置复用key。
- alias仅用于两个连续角色，绑定位置是输入Sequence展开后的0和2。
  非alias输入逐值检查未被修改；alias输入检查更新后的数值，额外加倍会被检测。

1个integration test通过，内部覆盖8次launch、24个输出张量。
测试源码见[foreach_sequence.rs](foreach_sequence.rs)，命令、源码/二进制/日志哈希见
[validation.json](validation.json)。构建使用绝对beta cargo/rustc、禁用wrapper和测试runner，
全部源文件与target均位于当前Burn仓库内，未改变生产依赖或Cargo缓存。

第二轮review发现旧测试的history均为`i + 1`，无法发现全部误用`history[0]`的缺陷。
修复数据后，在target副本故意把`history.index(tensor)`替换成`history.index(0usize)`：
测试成功编译，但在第二个张量的数值断言失败（首元素实际201，期望1201），退出码101。
随后恢复正确源码并重新通过测试；运行结果见[test-result.txt](test-result.txt)与
[mutation-result.txt](mutation-result.txt)。连续审查通过计数在本次修复后重新开始。
修复后的同一源码与证据已由Mill、Plato分别完成一轮只读agent review，连续2/2 PASS；
两轮均核验源码及归档证据，未将review当作额外执行测试。见[审查记录](../review-log.md)。

## 复现

先按[getter前置](../prerequisites/README.md)将指定CubeCL revision展开到仓库target副本。
本测试不调用新getter，getter补丁是否存在不改变其参数形式结论；记录中仍保存了实际runtime源码哈希。
从Burn根目录执行：

```sh
cp docs/optim-adamw/prototypes/foreach_sequence.rs \
  target/adamw-cubecl-stream-prerequisite/crates/cubecl-cpu/tests/foreach_sequence.rs
env -u CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER \
  RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= RUSTC=<beta>/bin/rustc \
  CARGO_TARGET_DIR=target/adamw-stream-prerequisite-build \
  <beta>/bin/cargo test --manifest-path target/adamw-cubecl-stream-prerequisite/Cargo.toml \
    -p cubecl-cpu --no-default-features --features std --test foreach_sequence
```

编译期展开循环中，optional Sequence使用`history.clone()`匹配；这仅复制编译期表示，
避免移动非Copy类型后下一次展开不可用，不是device tensor数据复制。
参数数目及chunk描述符参与编译参数，未来生产实现仍必须设置launch/展开上限并测量JIT。

## 尚未证明

本原型不证明AdamW公式、AMSGrad、runtime绑定预算、完整COW准入、不同有效stream的分组/恢复、
两次ModuleMapper、参数组/lr/time、checkpoint或GPU行为，也不提供加速比。
生产PR3仍受公开stream getter依赖接入与此前申请的额外fork发布确认限制。
