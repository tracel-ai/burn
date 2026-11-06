# AdamW 优化审查记录

本记录只计有可核实结果的 agent-generated review，不代表人工/维护者 review。代码修复后连续通过
计数重新从零开始；中断、API失败或源码在审查中变化的轮次不算通过。后续全部实现还需连续三轮总审查。

## PR1

| 审查 | 结论 | 发现与处理 |
|---|---|---|
| 初轮 | FAIL | 特殊值测试走回退；多步轨迹共享参数；内存proof污染峰值；Divan未正确预热/同步。已修复。 |
| Carson 修复复审 | PASS | 首次通过，随后发现新问题而失效。 |
| Peirce 独立审查 | FAIL | AMSGrad binding预算未检查；最终update乘法FMA收缩改变中间溢出分类。已修复。 |
| Confucius 审查 | FAIL | 低binding预算设备正确回退时测试误报launch不足。计数已按具体变体需求修复。 |
| Confucius 修复复审 | PASS | 当前版本第1轮，核对数值、布局、回退、feature、预算和观测隔离。 |
| Mencius 独立复审 | PASS | 当前版本第2轮，另复算4条runtime的报告数据，未发现新问题。 |

最终两轮共同审查的源码 SHA-256 前缀：optimizer/tests `aa44466b31bb`，kernel `5686d5461d36`。
CPU、Metal、WGPU/MSL、WGPU/WGSL、Fusion的AdamW测试各29项通过；raw CPU clippy通过。
no_std与burn-cubecl raw单测的既有编译失败如实记录在[报告](pr1-fused-report.md)和
[验证记录](data/pr1-validation.json)，未视作通过。报告性能数据来自每条runtime独立进程，保留原始样本。

审查agent只读；先前工作区并发修改来源不作为任何审查通过的证据。用户要求后续全部编码在当前仓库，
临时快照已停止使用，新增回归已合回 `/Volumes/HIKSEMI/code/burn`。

## 后续阶段

- PR2：实现及报告已完成，Hypatia初审PASS后补充capture回归；最终含该回归版本由Faraday、Hypatia连续两轮PASS。CPU/MSL/WGSL/Fusion各35项测试、Metal34项加capture1项通过，clippy通过。峰值内存改善成立；共享CPU tiny的速度不确定性明确列于[报告](pr2-cow-report.md)，未声称所有场景无回退。
- PR3：Kant、Euler连续两轮PASS确认当前pin缺少合格的有效stream查询API；[停止报告](pr3-foreach-report.md)已完成，Foreach实现仍阻塞，不能记为优化完成。
- PR4：早期cached binary/source归属冲突导致错误停止结论，旧测量已作废。禁用wrapper并用绝对toolchain直接重建后，CPU大参数有明确收益；当前实现CPU-only vector和单线程scalar tail，GPU保留scalar。新版本review进行中，尚无最终连续两轮通过。
- 最终三轮整体review：尚未开始。
