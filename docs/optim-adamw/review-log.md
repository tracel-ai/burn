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

## PR3 前置补丁增量

生产代码及依赖pin未变。重新检查官方head后确认仍缺getter，在本仓库target副本准备
[execution stream最小补丁](prerequisites/README.md)，两个CPU integration tests通过。
James、Ptolemy分别完成一轮独立只读review，连续2/2 PASS；James同次调用中的额外复核不另计一轮。
格式、干净pin apply检查通过；clippy失败于未改动的CubeCL dispatcher第69行，已记录。
此增量只是可接入上游的前置补丁和证据，不能计为PR3优化完成，也不改变此前生产实现3/3审查结论。

## PR3 Sequence 原型增量

生产代码及依赖pin未变。原型在本仓库target副本实际编译并执行，验证设计第8.2节的
Sequence参数形式，记录见[原型材料](prototypes/README.md)。

| 轮次 | reviewer | 结论 | 发现与处理 |
|---|---|---|---|
| S0 | Russell | PASS | 初始版本审查；后续发现测试盲点，该通过不计入最终连续通过。 |
| S1 | Mill | FAIL | history在各张量上均为`i+1`，全部误读history[0]也能通过。已加张量编号。 |
| S2 | Mill | PASS | 修复后第1轮，确认错绑变异编译成功后数值失败、正确源码复测通过，证据hash一致。 |
| S3 | Plato | PASS | 修复后第2轮，独立复核bounds、stride、chunk覆盖、alias绑定、key声明及全部证据hash。 |

最终连续 **2/2 PASS**，对应源码SHA-256为
`0a6813e8ea0e461bde577dbfdb1e17dd1a14abb81c63a48b2c76924d82cddfae`。
正确版本1个integration test通过，含8次launch、24个输出张量；故障注入退出101并在数值断言失败。
两轮review均为只读，没有将核验已有运行记录描述为再次运行。此增量不是AdamW算法实现、性能优化
或PR3验收；Foreach的生产依赖前置与完整实现仍未完成。

## PR3 本地实现候选增量

依赖发布确认尚未到达期间，在当前仓库target中的getter补丁依赖上完成CPU Foreach本地实现、
测试和测量，以[可复现候选补丁与报告](experiments/pr3-local/README.md)归档。
本增量恢复生产源码及原pin，只提交归档材料，不把root path patch当作可发布依赖方案。

| 审查 | reviewer | 结论 | 范围 |
|---|---|---|---|
| 中间模块审查 | Dalton | PASS | 动态分派、两次遍历、state/context和回退；指出parameter COW额外成本并纳入报告 |
| 中间kernel审查 | Meitner | PASS | candidate2；建议补非恒定索引值、chunk累计超限、AMSGrad迁移、跨项alias和第三stream恢复，已补 |
| 正式候选第1轮 | Meitner | PASS | candidate4代码、bench和tests；CPU-only、共同COW、stream、分组和回填，未发现确认缺陷 |
| 正式候选第2轮 | Feynman | PASS | candidate4完整代码及归档；内存重放patch、源码/样本/日志hash、复现脚本产物和报告数值核对通过 |

正式本地候选连续 **2/2 PASS**，主补丁SHA-256
`cb6e03726063d1534976d404d32c209cd95f77e23ec0c6d3a4739bb81ecfbd11`。
两轮均只读，不把核验既有测试记录描述为额外执行。代码测试、性能限制、GPU未启用的依据和
完整源码hash见候选报告及validation。此通过仅许可归档候选，不是PR3生产交付或最终全分支验收。

## PR3 候选修复与验收补充

[修复报告](experiments/pr3-local/refinement/README.md)记录占满chunk预算时提前回退、
near-limit pool峰值28→24MiB、数值判别用例、host阶段计时及真实Group/Capture回退验证。
原candidate4补丁和数据保留，修复作为后续patch应用。

| 轮次 | reviewer | 结论 | 范围与处理 |
|---|---|---|---|
| R0 代码修复 | Halley | PASS | 预算边界、COW、特殊值、rank8迁移及benchmark；尚未包含decorator包 |
| R0 补充harness | Halley | FAIL | 只比较两模式的checkpoint策略，不能发现两者同时恢复成默认值；改为分别对照初始非Disabled策略，重新测试和审查 |
| R1 修复复审 | Halley | PASS | 完整稳定修复增量与checks；独立checkpoint策略断言及重跑证据核验，确认原P2关闭 |
| R2 独立复审 | Feynman | PASS | 全部增量、patch重放、9个源码hash、61项数据、checks12项hash、日志与报告算术核验，无新增确认问题 |

失败后重新计数，最终连续 **2/2 PASS**；两轮均为只读review，没有重新执行测试。
refinement补丁SHA-256为`66da8d1421756123d1ee1aeb9db097c725ccb949322cf2a2d0b7c6de3fb0130c`。
最终记录CPU49pass/2ignored、Fusion13pass/1ignored、clippy通过，以及实际Group/Capture2pass；
CPU专用用例在Fusion提前返回，Capture验证IR录制，不扩大覆盖声明。本增量继续以patch归档，
生产依赖发布、正式接入和全分支最终审查仍未完成。

## PR3 Remote 与独立下游补充

[Remote验证包](experiments/pr3-local/checks/README.md)新增真实loopback WebSocket CPU服务端测试；
[独立下游](prerequisites/downstream/README.md)验证本地git依赖不依赖根patch即可编译和实际batch执行。
本轮不修改已审查的生产候选，只扩充验证材料。

| 轮次 | reviewer | 结论 | 范围与处理 |
|---|---|---|---|
| 中间Remote审查 | Archimedes | FAIL | 阻塞read/sync无整体超时；按既有Remote测试加入30秒worker watchdog并重跑 |
| 中间修复复核 | Archimedes | PASS | 关闭watchdog问题；脚本默认Bash空数组问题已实际修复、默认入口复测通过，不计正式连续轮次 |
| 正式第1轮 | Archimedes | PASS | Remote、localgit依赖图与runtime、证据hash、脚本边界和报告范围整体审查 |
| 正式第2轮 | Schrodinger | PASS | 独立核对Remote、17项checks与246项downstream hash、实际fetch源码、无根覆盖和数值证据；指出一处旧探针描述，已改为本轮重跑事实 |

首次下游consumer因模块参数未启用autodiff而runtime失败，修正consumer后通过，原始失败仍保留。
未修改候选算法来迎合测试，也未把compile-only或失败轮次计为实际执行通过。
正式连续 **2/2 PASS**；两轮均只读，未重跑构建或测试。此增量验证本地方案和回退边界，
不能替代额外仓库发布、生产pin同步或最终整体验收。
