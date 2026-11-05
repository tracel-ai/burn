# AdamW 性能优化 PR 路线

> 状态：设计稿，尚未实现或测量
>
> 基线：Burn `ee39002d94e1c958d86a65e92132b3961d48c18c`
>
> CubeCL：`d88b0febb50e7b83a8548125f6a701c12f11a2ba`（`0.12.0-pre.2`）
>
> 关联文档：[CubeCL CPU 运行时优化](optim-adamw-cpu-runtime-optimization.md)

本文只负责路线导航和 PR 划分。每项优化各有一份**自包含设计文档**，可以单独阅读、评审和实施，不需要先阅读其他 PR 文档。

## 1. PR 文档

| PR | 独立设计文档 | 性能目标 | 代码依赖 |
|---|---|---|---|
| PR 1：单参数 out-of-place Fused | [pr1-fused.md](optim-adamw/pr1-fused.md) | 减少单参数 AdamW 的逻辑中间 tensor、算子分发和物理 launch | 无，可基于 `main` |
| PR 2：参数与 state 的 COW 输出复用 | [pr2-cow.md](optim-adamw/pr2-cow.md) | 减少稳态输出分配、回收和峰值内存 | PR 1 已合并 |
| PR 3：模块级多参数 Foreach | [pr3-foreach.md](optim-adamw/pr3-foreach.md) | 减少多参数模型的 launch 与 host enqueue 开销 | PR 1 已合并；CubeCL 有效 stream 查询；PR 2 可选 |
| PR 4：大参数吞吐优化 | [pr4-throughput.md](optim-adamw/pr4-throughput.md) | 提高大参数内核有效吞吐，并在有收益时扩大快路径范围 | PR 1 已合并 |

每份 PR 文档都独立包含：

- 性能问题、直接对照组和适用 workload；
- 实现范围及明确不做的内容；
- 本 PR 需要遵守的数值、状态和布局契约；
- 正确性测试、路径观测和验证命令；
- benchmark 方法、性能验收和停止条件。

## 2. 划分原则

每个 PR 必须形成完整的性能闭环：

1. 说明一个明确的性能问题；
2. 携带复现问题和验证改进所需的 benchmark；
3. 实现完整可用的优化，不留待后续 PR 补齐的生产骨架；
4. 通过本 PR 的正确性、兼容性和路径验证；
5. 用同机、同配置 A/B 证明本 PR 自身有统计上明确的净收益；
6. 单独更新 rustdoc、changelog 和 PR 描述，只声明实际测得的结果。

以下工作可以并入性能 PR，但不单独成为 PR：

- Reference 机械整理；
- 配置兼容测试；
- benchmark harness 和观测计数器；
- compile-only prototype；
- 没有实测收益的常量调优。

正确性是合并门禁，但不是合并无性能收益改动的理由。实现正确却没有净收益时，不提交或不合并该 PR。

## 3. 依赖不等于捆绑交付

- 无代码依赖的 PR 可以直接基于最新 `main` 开发。
- 有代码依赖的 PR 只依赖**已经合并**的前序实现，不形成必须整体合入的未合并 PR 链。
- 每个 PR 使用自己的直接对照组：PR 1 对 Reference；PR 2 对 out-of-place Fused；PR 3 对当前逐参数 Fused；PR 4 对合入前的大参数 Fused。
- PR 2 被跳过时，PR 3 仍可用 out-of-place 输出独立推进；PR 4 不依赖 PR 2/3。
- PR 编号是稳定标识，不要求机械地按 1→2→3→4 执行。PR 1 合并后，根据 profile 选择下一项。
- 两个 PR 修改同一内核时，后合入者 rebase 后重新验证性能和正确性。
- 任一 PR 可以单独回退，不影响无关优化。
- 文档自包含与代码依赖是两件事：即使实现依赖 PR 1，PR 2/3/4 的文档仍完整说明自己的约束和验收方法。

一次只活跃一个实现 PR。本地 spike 可以用于排除不可行方案，但不作为待合并 PR。

## 4. 共同边界

各 PR 文档会按自己的范围重复列出相关约束，以下仅作路线摘要：

- 现有 `AdamWConfig` 默认值和 Reference 算法保持原样；只有显式选择新执行方式才尝试专用路径。
- 不扩大核心 `Backend` trait；不支持的 backend、dtype、layout 和执行模式回退 Reference。
- Fused 融合单参数内的纵向数据流；Foreach 批处理多个独立参数，不复制或拼接整个模型。
- 不实现全局梯度裁剪、不引入 FP32 master weights、不改变逐参数 clipping、不改变 checkpoint 格式。
- `cubecl-cpu` 调度器和内存回收的上游改动不属于这些 Burn PR 的隐含前置条件。
- 不承诺全模型固定 launch 数、所有设备走专用内核、bitwise 一致或“零分配”。
- 首版专用路径只开放 F32 非量化输入；半精度开放需要另一个完整数值设计。
- 性能结论只覆盖实际测量的 hardware、runtime、feature 和 workload。

## 5. 推进顺序

1. 先独立完成 PR 1，证明单参数 Fused 的端到端净收益。
2. PR 1 合并后查看 profile：
   - 分配/drop 显著：推进 PR 2；
   - 多参数 launch/host 显著：推进 PR 3；
   - 大参数吞吐不足：推进 PR 4。
3. 每个后续 PR 只与包含前序优化、但不包含自身改动的直接基线比较。
4. 没有对应瓶颈或没有净收益的可选 PR 直接跳过，不为了完成路线而合并。
