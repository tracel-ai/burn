# PR3 候选补充：不可合批边界、数值验收和耗时分项

状态：**本地候选的后续修复，仍未接入生产依赖，不计PR3交付。**
先应用[原候选补丁](../foreach.patch)，再应用本目录的
[candidate-refinement.patch](candidate-refinement.patch)。原候选及原测量文件保留不变。

## 修复占满chunk预算的参数

原候选允许单参数恰好使用全部32个chunk。这样的参数不可能与另一个非空张量合批，却仍进入
模块收集/写回流程。收集阶段保留旧参数，破坏其独占条件，使最终singleton Fused也无法复用参数buffer。

准入现改为`ceil(numel / chunk_elements) < max_chunks`。占满预算时，立即返回未消费的输入，
模块释放额外保留句柄后执行原Fused。它没有移除任何能与另一参数合批的情况：
`31*4096`元素的张量与1元素张量仍能共同占用32个chunk并提交一次batch。

新增边界测试检查`32*4096-1`、`32*4096`、`32*4096+1`的拒绝与allocation保持，
并实际验证`31chunks+1元素`仍为一次batch。模块测试连续两步确认`[512,256]`参数保持原allocation，
每步single=1、batch=0，参数数值与逐参数Fused一致。

独立进程内存复测结果：

| near_limit：8×[512,256] | 原候选Foreach | 修复后Foreach | Fused |
|---|---:|---:|---:|
| pool峰值之和（MiB） | 28 | 24 | 24 |
| 稳态live allocation | 24 | 24 | 24 |
| 稳态live bytes（MiB） | 12 | 12 | 12 |
| batch / single | 0 / 8 | 0 / 8 | 0 / 8 |

pool峰值之和仍是上界。这里直接修复了可确认的多余参数分配；没有据此声称所有singleton都能复用，
不同lr/time等原因产生的singleton仍可能经过收集。

## 补足数值区分力

- AMSGrad特殊值以两个真实batch项运行，第二项调换槽位：旧max为NaN而新m2有限时保留NaN；
  新m2为NaN而旧max为7时保留7；旧max为负零与新m2正零相等时保留负零位模式。
  同时对照Reference的参数、m1、m2及max，断言batch=1、single=0。
- 开关迁移升级为rank1/rank8模块，每步保存并重新加载checkpoint，执行模式交替Reference/Foreach。
  `beta_1=beta_2=0.5`、梯度依次`4,2,1,0.5`、AMSGrad依次`false,true,false,true`。
  m2应为`8,6,3.5,1.875`，max从第2步起始终为6，直接区分“重新启用时忽略旧max”等错误。
  每项检查time、`__rank`、state名称、shape、dtype、path，并在Foreach两步断言真实batch。
- benchmark路径检查由“有任意optimizer launch”加强为各workload实际batch/single数量断言，
  防止整个Foreach回退仍被当作batch性能。

当前CPU AdamW检查为**49 passed、2 ignored**；Fusion新增用例为**13 passed、1 ignored**，
其中CPU专用用例按能力返回，不能计为Fusion batch验证。raw CPU all-targets clippy通过。
本次没有改变JIT key生成，两个专项JIT的上一阶段结果保留；不把它们写成本次复跑。
日志摘要及哈希见[validation.json](validation.json)和`validation/`。

另用公开API的[独立decorator验证包](../checks/README.md)实际通过2项测试，均无early-return：
Group两参数两步及checkpoint恢复的数值、全部state、device和context一致，AdamW专用launch为零；
`Device::capture()`对两参数两步录制出62个相同的规范化IR操作与8个输出，所有raw launch为零。
observer先以raw CPU single launch作正向校准。Capture只证明IR录制语义，不是数值执行；
Remote仍未验证。测试包独立锁文件、完整IR和运行摘要均已归档，不改变生产feature。

## 修复后的配对性能

环境、绝对Rust工具链和禁用wrapper方法沿用原报告。每进程两模式交替执行40组，
各预热8步、每步前后sync，重复3个进程，采样时无本agent并行构建。
参数与梯度构造在计时区外；两次模块遍历、metadata、state、分配、提交均在step内。
下表耗时是三轮各自中位数的中位数，单位ms；比值为各轮Fused中位数/Foreach中位数。

| workload | Fused total | Foreach total | 三轮比值 |
|---|---:|---:|---|
| near_limit | 0.9219 | 1.1650 | 1.034 / 1.125 / 0.711 |
| tiny | 1.0303 | 0.6973 | 1.330 / 1.668 / 1.380 |
| long_short | 2.0658 | 1.8547 | 1.328 / 1.000 / 1.111 |
| large_1 | 0.4349 | 0.4347 | 0.959 / 1.006 / 1.010 |
| large_4 | 2.1805 | 2.2945 | 1.019 / 0.913 / 1.067 |
| large_16 | 9.4403 | 10.9298 | 0.731 / 0.992 / 0.989 |
| over_limit | 8.6362 | 1.9937 | 3.501 / 4.332 / 5.356 |

tiny与over_limit三轮均有净收益。near_limit内存修复成立，但第三轮耗时仍明显回退；large_16同样
存在回退样本。总耗时不能被解释为“所有场景无退化”，正式性能门禁仍需对这些场景继续评估。
测试覆盖在benchmark构建后继续加强，但最终生产代码与bench源码hash均与实测binary一致；
仅测试源码不同，差异明确保存在validation中，没有覆盖原测量hash。

## Host阶段测量

另用[临时诊断补丁](profile/phase-profile.patch)分别计时真实执行路径，未增加生产feature或观测API。
补丁基于本次生产/bench版本，应用于本仓库源码后编译独立binary，完成后恢复原源码。
每workload先首步及5次预热，再路径步，随后30个稳态样本；表中为30个样本各字段中位数（ms）。
每步多次batch的字段先相加，再取中位数。

| Foreach workload | 收集/即时Fused | 分组 | COW与输出分配 | views/chunks | 提交 | state/context恢复 | 第二遍写回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 0.07973 | 0.00550 | 0.03233 | 0.01869 | 0.05285 | 0.02394 | 0.00879 |
| mixed_rank | 0.00929 | 0.00071 | 0.00354 | 0.00275 | 0.00729 | 0.00304 | 0.00142 |
| long_short | 0.10404 | 0.00679 | 0.03138 | 0.01498 | 0.05214 | 0.02590 | 0.00910 |
| over_limit | 0.19262 | 0.01188 | 0.07396 | 0.03831 | 0.12638 | 0.05912 | 0.01900 |
| near_limit | 0.07765 | 0 | 0 | 0 | 0 | 0 | 0 |

Fused的一遍module map包含实际逐参数提交，其中位数tiny为0.21706ms、mixed_rank为0.02517ms、
long_short为0.22810ms、over_limit为8.89681ms、near_limit为0.06187ms。
Foreach收集阶段也包含不合格/过大项的即时Fused，例如long_short的两个大矩阵；不是纯metadata计时。
分组不包含后续launch拆分的所有控制开销。字段中位数不能相加当作整体中位数。

这些真实分项比之前identity-map下界更接近开销来源，但每次`Instant`与诊断输出会扰动执行，
尤其enqueue/backpressure；仅用于定位成本，不能与无计时代码的吞吐表混用或相减得到device时间。
临时源码patch、hash、binary hash及原始stderr/stdout全部在`profile/`，最终候选不含该计时代码。

## Profiling窗口的实际方法

benchmark新增`profile-fused`/`profile-foreach`命令，使用现有`Device::profile`并打印
`ProfileDuration::timing_method()`。本机CPU实际全部返回 **system**，每模式每workload采20个窗口。

| workload | Fused窗口中位数ms | Foreach窗口中位数ms | method |
|---|---:|---:|---|
| near_limit | 1.31377 | 1.05106 | system |
| tiny | 1.59896 | 0.63925 | system |
| long_short | 1.96719 | 1.87419 | system |
| over_limit | 7.40381 | 2.06575 | system |

该结果是同步边界间的系统时间，不能称为纯device/kernel耗时。吞吐与系统profile分开保留；
当前后端没有通过这条公开API提供本次测量可用的独立device时间，仍是报告限制。

## 使用与仍未完成的交付条件

```sh
git apply docs/optim-adamw/experiments/pr3-local/foreach.patch
git apply docs/optim-adamw/experiments/pr3-local/refinement/candidate-refinement.patch
# 按父报告准备同一CubeCL本地补丁，使用相同absolute toolchain和--config构建。
# <bench-binary> --module-samples paired near_limit
# <bench-binary> --module-samples foreach near_limit
# <bench-binary> --module-samples profile-foreach tiny
```

该增量修复内存问题并补强证据，未绕过远程getter、CubeK/Burn pin同步、下游依赖图验证、
正式性能门禁、PR3连续两轮和最终分支连续三轮review。生产依赖仍未改动。

该本地增量已在修复checkpoint策略测试盲点后，由Halley、Feynman连续两轮独立agent review通过；
失败轮次未计入通过。范围、发现及证据见[审查记录](../../../review-log.md)，不代表生产PR3验收。
