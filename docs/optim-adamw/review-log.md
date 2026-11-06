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
- PR4：旧cached binary/source冲突已隔离；禁用wrapper的直接构建证实CPU-only向量化和serial tail收益。修复报告基线标记及首步/变体/体积证据后，Darwin、Sagan对完整最终版本连续两轮PASS。kernel `da4cb826519f`、optimizer `442ec4a483f2`、bench `b5c972335bb1`。CPU/Metal/MSL/WGSL/Fusion各36项测试通过，CPU专项JIT通过，直接构建raw CPU all-targets clippy通过。
- 阶段提交：用户已将生产代码提交于[用户的生产实现提交](https://github.com/onenewcode/burn/commit/ba84da059c08bd905a5c89acc863778e8e928b1b)；本阶段报告及验证证据随后单独提交并推送。
- 最终整体审查G0（Hegel）：FAIL，公共rustdoc将未验证的浏览器WebGPU列成已验证；已改为仅CPU/Metal/native WGPU-WGSL/MSL已验证，其余明确未验证。生产kernel和性能样本未变化，重新计数0/3。

## 最终整体审查

| 轮次 | reviewer | 结论 | 范围与结果 |
|---|---|---|---|
| G0 | Hegel | FAIL | 公开rustdoc误将浏览器WebGPU列为已验证；修正文档后重新计数。 |
| G1 | Hegel | PASS | 完整分支审查及G0修复复核，生产hash与测量证据一致。 |
| G2 | Noether | PASS | 独立核查数值、COW、vector/tail、回退、feature与报告，无新增确认缺陷。 |
| G3 | Turing | PASS | 独立完整审查及源码/二进制/验证日志核查，无新增确认缺陷。 |

最终连续计数 **3/3 PASS**。审查对象为基础设计提交至阶段提交及最终文档修正；
审查时生产hash：kernel `da4cb826519f`、optimizer `442ec4a483f2`、bench `b5c972335bb1`、bridge `a985edb26cf27`。
候选binary `e24b0fb99644`、PR4 scalar baseline `f9bf42b7d274` 的实际hash与manifest一致。
最终提交前rustfmt将一个if表达式换行，kernel hash变为`56638a88be99`；去除空白后的源码一致，
没有行为变更。原始测量hash未重写，映射见[final-formatting.json](data/final-formatting.json)。

阶段报告和证据已提交并推送为`a5e263564`；本最终记录与WebGPU文档修正将再提交推送。
该三轮通过仅说明当前分支实现及材料通过agent review；PR3 Foreach仍受上游getter阻塞，
不能据此将“全部PR实现”的用户目标标记为完成。
