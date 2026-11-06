# PR3：Foreach 前置能力审查与停止报告

状态：**受上游 API 前置阻塞，尚未实现 Foreach，不计作完成的优化。**
当前 Burn 锁定 CubeCL `d88b0febb50e7b83a8548125f6a701c12f11a2ba`。
本文依据 [PR3 设计第6节](pr3-foreach.md#6-有效执行-stream-是启用门禁) 的明确启用门禁，记录可核实结论。

## 阻塞证据

CubeCL `crates/cubecl-runtime/src/client.rs` 的 `Client::stream_id()` 为私有方法，内部规则为：

```rust
match self.stream_id {
    Some(pinned) => pinned,
    None => StreamId::current(),
}
```

`Client::service_id()` 已公开；`service_stream()` 和 execution 模块的 `ServiceStream` 仍为
`pub(crate)`。公开的 `unsafe set_stream()` 能设置 client，却不能读取原来的 pinned 状态。
`ProfileWindow.stream_id` 虽然公开，取得它的 `profile_start()` 会向 server 发送阻塞请求并打开
profiling window，不能充当设计要求的无副作用只读查询。

`Handle.stream` 记录 allocation 的创建 stream，不能表示本次 client 的有效执行 stream。
只使用 `StreamId::current()` 会丢失 pinned client 的语义；把所有 client 改成当前 stream 也会改变
原本的执行顺序。安全的 Foreach 收集必须在原逐参数执行作用域记录有效 stream，并在每组分配输出、
launch及延迟回退时恢复同一作用域。

## 最小前置与后续工作

上游可以提供等价只读 getter，例如：

```rust
/// The stream used for this client's next operation, including an explicit pin.
pub fn execution_stream(&self) -> StreamId {
    self.stream_id()
}
```

该方法应随锁定的 CubeCL revision 提供，并覆盖 pinned/unpinned 及嵌套 stream scope 测试。
没有修改 Cargo 缓存、猜测私有布局或用 allocation stream 冒充执行 stream。
前置满足后仍需完成 Sequence 编译原型、完整分组键、chunk上限、两次ModuleMapper接入、参数组与lr/time保持、
回退/恢复stream、JIT与端到端A/B，随后重新完成连续两轮review。只增加 getter 不算实现 PR3。

## 性能结论

| 项目 | 当前行为 | PR3 后行为 |
|---|---|---|
| 执行配置 | Reference / 单参数 Fused | 未增加 Foreach variant |
| 单参数 Fused launch | 每参数1次 | 未改变 |
| 多参数 batching | 未实现 | 未启用 |
| 性能数据 | PR1/PR2报告 | 没有 Foreach 实测，不提供虚构加速比 |

设计第13节明确要求“无法查询并恢复有效 stream”时停止该 PR。因此后续独立吞吐优化可以继续，
但总目标中的 Foreach 实现仍缺失，不能把该停止报告当作“全部 PR 已实现”的证明。
本判断由编码 agent 生成；Kant、Euler 连续两轮独立源码审查均确认停止条件成立。
这两次 PASS 不算 Foreach 实现或性能验收，记录见 [review-log.md](review-log.md)。

## 前置补丁验证

重新检查官方 head `565108e077557f634aa7f5bf19b34ed655600ba7`（比pin新16个提交），
查询入口仍未公开，直接升级到该版本不能解除门禁。已在当前Burn仓库的`target/`副本中实现
最小`execution_stream()` getter，并在CPU上通过两个integration tests，覆盖pinned/unpinned、
clone、嵌套scope及随后allocation实际stream。生产pin和Cargo缓存均未修改。

可复现补丁、哈希和验证命令见[前置补丁](prerequisites/README.md)。这是上游能力准备，
不算完成Foreach，也不替代本PR的数值、分组、stream恢复或端到端性能验收。
此外，当前CubeK也固定相同CubeCL revision；引入getter后需要同步依赖图，不能只更新Burn的pin。
