# CubeCL CPU 微任务调度优化：安全约束、实验设计与实施阶段

> 状态：设计稿，尚未实现或测量
> 基线：Burn `ee39002d94e1c958d86a65e92132b3961d48c18c`
> 上游 CubeCL：`d88b0febb50e7b83a8548125f6a701c12f11a2ba`
> 关联文档：[AdamW Fused/Foreach 方案](optim-adamw-universal-optimization.md)

## 0. 文档目的与结论

这份文档只讨论 `cubecl-cpu` 运行时，不再把未复现的 60～80 ms 数据当作已确认根因，也不把 AdamW 算法融合与 CPU 线程池改造捆成同一个 PR。

建议按以下顺序推进：

1. **先测**：分解 Host enqueue、Worker 执行、stream drain、资源 drop 和首次 JIT。
2. **先减少工作**：AdamW Fused/Foreach 在上层减少不必要的任务和中间 tensor。
3. **受限 inline**：只对没有 barrier、没有 shared-memory 生命周期问题、没有未完成依赖且工作量足够小的单 unit 任务绕过全局线程池锁和 channel。
4. **批投递后评估**：先找到 scheduler 的真实批边界，再评估“一次锁、多项任务”，不预设 99% 收益。
5. **内存回收最后处理**：当前源码不支持“全局内存池互斥锁是已证实主因”的说法；没有 profiler 证据和所有权设计，不引入后台 free-list。

默认原则是：不能证明并发安全或没有可复现净收益的改动不进入默认路径。

## 1. 锁定版本的实际执行路径

### 1.1 全局锁存在，但不能据此推出锁竞争就是主因

`cubecl-cpu/src/compute/threadpool/mod.rs` 当前有：

```rust
static INSTANCE: OnceLock<spin::Mutex<Threadpool>> = OnceLock::new();
```

`CpuStream::enqueue_task` 对每个 Execute 调用：

```rust
self.threadpool.lock().execute_data(...);
```

因此，**每次执行会获取一次全局 Threadpool 锁**是源码事实。是否存在明显争用、持锁时间多长、是否阻塞其他 stream，则必须测量；单线程连续 launch 也可能主要支付的是任务准备、channel、JIT 查询或同步成本，而不是锁竞争。

### 1.2 `units` 不是元素数

当前 Host 计算：

```rust
let units = cube_dim.num_elems();
```

`Threadpool::execute_data` 遍历的是 `cube_dim.x/y/z`，为每个 unit 创建一个 `ComputeTask`。`cube_count` 被写入 `PlironData.builtins` 交给 JIT kernel，不是这里的 task 循环维度。

所以原稿的“Units < 4096 即是微任务”没有定义清楚：

- `cube_dim.num_elems()` 决定本次创建多少 task；
- `cube_count` 描述 grid，但不同 kernel 的每 unit 指令数、向量宽度和循环仍不同；
- 不能把 task 数直接当作元素数或纳秒级成本。

工作量估计最多先把 `cube_count.x * cube_count.y * cube_count.z` 当作粗粒度 proxy，再用实测校准，不能宣称是精确 FLOPs。

### 1.3 Dispatcher 的真实行为

当前 `DispatcherScheduler`：

- 对普通 task，根据各 Worker 的原子队列长度选择相对空闲者；
- 对 `needs_parallelism` 的 task，按 unit index 绑定固定 Worker；
- 每个 task 仍通过 `std::sync::mpsc::channel` 发送；
- 队列长度使用 `CachePadded<AtomicUsize>`，不能未经证据就宣称存在特定 false-sharing 热点。

`DispatcherWorker::work` 的行为是：

1. 在 `IDLE_POLL = 200µs` 内 `try_recv`；
2. 未收到时 `yield_now`；
3. 超过预算后调用阻塞 `recv`，不是永久忙等；
4. 未 ready 的 task 放入 `aside`，ready 后执行并递减队列长度。

当前源码注释明确说明 polling/yield 是在 Worker 覆盖所有逻辑 CPU 时避免饿死 client 和其他 unit。是否应调整必须同时测量尾延迟和 idle CPU，不能只数 `yield_now` 就断定上下文切换占据主要时间。

### 1.4 Stream 顺序、counter 和 keepalive 是安全核心

每个 `CpuStream` 有：

```text
next_counter_step
atomic_counter
main MemoryManagement
shared_memory_management
threadpool: &'static spin::Mutex<Threadpool>
```

`ComputeTask`：

- `is_ready()` 要求 `atomic_counter >= next_counter_step`；
- `compute()` 执行 `PlironEngine::run_kernel`，随后 `fetch_add(1, Release)`；
- Host 在 launch 后增加 `next_counter_step`；
- `submit()` 自旋后 yield，直到 counter 追上目标。

`execute_data` 还把 resource 装入 `PlironData.shared.keepalive`，让已排队/运行中的 kernel 持有绑定。任何 inline 或异步回收都必须保留：

- 同流旧任务先于新任务的顺序；
- counter 的精确增量和 Acquire/Release 配对；
- 跨 stream 对齐关系；
- 输入、输出和 metadata 在整个执行期间的生命周期。

### 1.5 Barrier 与 shared memory 不能走简单串行直通

当前代码对 `requirements.needs_parallelism` 调用 `ensure_workers(cube_dim.num_elems())`，因为 cube barrier 要求一个 cube 的所有 unit 都在运行。把多 unit barrier kernel 在调用线程顺序执行可能死锁。

Shared-memory launch 还有另一约束：当前在 enqueue 前先 `submit()`， reservation 在 enqueue 后释放，因此同一 shared pool 不与未完成的 shared-memory launch 重叠。简单 inline 若改变这个边界，可能让下一 launch 复用仍在读取的内存。

第一版 inline 同时排除这两类，而不是只检查 task 大小。

### 1.6 内存管理不是当前源码所描述的“全局 Mutex 池”

`MemoryManagement` 的字段是 per-stream 的 persistent/dedicated/dynamic pools 和 storage，并由 `&mut self` 驱动。没有证据支持原稿的“全局 `MemoryManagement` 互斥锁 + BTreeMap 引发锁风暴”归因。

这并不证明 drop 没有成本。可能的成本包括：

- `SharedData.keepalive` 中多个 Box/resource 的析构；
- stream 调度与绑定 bookkeeping；
- pool coalesce/cleanup；
- `submit()` 等待任务结束；
- 实际系统 allocator 的释放。

必须通过独立计时和 profiler 区分，不能预先命名为锁问题。

## 2. 原稿中不再采用的判断

| 原判断 | 修正 |
|---|---|
| Flex 调度开销恒等于 0 | Flex 仍有 tensor、COW、SIMD 和 allocator 成本，只是没有 CubeCL 这套虚拟 cube 调度。 |
| GPU 异步完全掩盖发射成本 | 异步只改变等待位置，Host enqueue、同步和驱动成本仍存在。 |
|  Batch 1 与 128 接近证明 optimizer 全是固定调度成本 | 纯 optimizer 的数学工作量在梯度 shape 固定时与 batch 无关；端到端差异还含前反向、内存和同步，不能据此定位根因。 |
| 450 个高阶算子就是 450 次物理 launch | Fusion、COW 和执行模式会改变物理数量，必须实测。 |
| inline 后单任务延迟必降到几十纳秒 | JIT 函数执行、资源准备和实际计算仍在，不能承诺纳秒数。 |
| 一次批投递把锁次数减少 99% | 取决于真实连续批长度；channel send 和 task 创建不会因一次锁消失。 |
| thread-local free-list 可以直接复用 drop 的 buffer | in-flight task 可能仍读取，且 buffer 有 stream/allocation 归属；没有 quiescence 就不能复用。 |

## 3. P0：先建立可复现测量

### 3.1 固定 workload

至少保留四类：

1. 单个小 elementwise kernel，验证固定 launch 成本；
2. 30 个 `[64,64]` F32 参数的 AdamW step，区分 Reference/Fused；
3. 大 `[1024,1024]` elementwise，验证 inline 不伤吞吐；
4. barrier、shared-memory 和多 stream 混合任务，验证正确性和尾延迟。

每个 workload 记录 tensor shape/dtype、`cube_dim`、`cube_count`、是否 `needs_parallelism`、shared blocks、bindings 数和 stream 数。

### 3.2 分段计时，不用一个总时间解释全部

建议增加只在 profiling feature 下启用的计数器：

```text
prepare_bindings
threadpool_lock_wait
execute_data_prepare
channel_send
worker_wait_before_ready
run_kernel
keepalive_drop
client_submit_wait
memory_cleanup
```

同时记录：

- 每类 task 数量和直方图；
- 每个 Worker 的 queue length；
- `try_recv` miss、yield、blocking recv 次数；
- inline eligible/selected/completed 数；
- OS voluntary context switch（平台可提供时）；
- idle pool CPU 使用率。

计数器不能本身成为全局热点：先用 per-stream/per-worker 局部计数，在 profiling 结束时汇总。

### 3.3 测量协议

- Release 和 Debug 分开，不能混表；
- 首次 LLVM JIT 单独报告；
- 稳态至少预热多轮，报告 median、p95、样本数和原始数据；
- Host-only 与 `Device::sync` 后总时间分开；
- drop 区与 compute 区分开；
- 记录 CPU 型号、核心/affinity、OS、电源状态、线程数和 Cargo features；
- A/B 使用同机、同 workload、同 locked revision；
- 对照至少含 Flex，但不得把后端差异全部归因于 Threadpool。

### 3.4 根因判定门槛

只有数据支持时才进入对应方案：

| 观测 | 才考虑的方案 |
|---|---|
| 单 unit 小任务中锁 + channel + worker 交接占显著比例 | 受限 inline |
| 连续多个 Execute 的重复锁获取占显著比例 | 批投递 |
| `yield_now`/OS context switch 与尾延迟相关 | polling 退避实验 |
| keepalive drop 或 cleanup 在 profile 中占显著比例 | 内存所有权/回收优化 |
| 主要是 JIT/实际 run_kernel | 改 kernel、融合或向量/指令优化，而不是调度器 |

## 4. P1：受限单任务 Inline

### 4.1 先拆分 launch 准备与调度

当前 `Threadpool::execute_data` 同时负责：

1. 提取 buffer pointers；
2. reserve shared memory；
3. 创建 `PlironData` 和 keepalive；
4. 创建每个 unit 的 `ComputeTask`；
5. 调度发送。

若仍在 `self.threadpool.lock()` 内决定 inline，只能减少 channel，不能绕过锁。因此重构为：

```text
PreparedLaunch::prepare(...)       // 不需要 &mut Threadpool
    ├─ inline_eligible + stream idle
    │      └─ PreparedLaunch::execute_inline(...)
    └─ otherwise
           └─ threadpool.lock().execute_prepared(...)
```

`PreparedLaunch` 持有 engine、bindings/resource、metadata、cube_dim/cube_count、base data 和 requirements。正常路径仍逐 unit 构造 task；单 unit inline 不需要创建 channel task 列表。

shared-memory reservation 在 prepare 阶段需要 `&mut CpuStream.shared_memory_management`，但第一版 inline 排除 shared memory，正常路径保持现有生命周期不变。

### 4.2 第一版全部准入条件

只有同时满足下列条件才允许 inline：

1. `cube_dim.num_elems() == 1`，本次只需要一个 unit task；
2. `!pliron_engine.requirements().needs_parallelism`；
3. `requirements.shared_memories.blocks.is_empty()`；
4. launch 已完成编译，dynamic cube count 已按现有路径解析；
5. `atomic_counter.load(Acquire) == next_counter_step`，同流没有未完成的旧任务；
6. 当前 launch 没有需要等待的跨 stream 对齐/读写依赖；
7. 工作量 proxy 小于经 benchmark 校准的阈值；
8. 当前不是 Discard/CompileOnly（这些模式本来就不应执行）；
9. 没有禁用 inline 的 debug/runtime 配置。

第 5 条故意选择“stream idle 才 inline”，而不是在 client 上忙等旧任务：如果旧任务未结束，直接走正常异步路径，避免越过旧任务或引入新的等待死锁。

第 6 条需要在 `server.launch` 已有 `streams_to_align`/`written` 信息处计算并传入 `ScheduleTask::Execute { allow_inline, ... }`。第一版可保守地要求所有读写 buffer 都属于当前 stream；不同 stream 的输入输出直接禁止 inline。

### 4.3 工作量阈值不是通用常数

实验阶段可以从 `cube_count_product <= T_experiment` 开始，但该值只用于 A/B，不直接作为跨平台默认值。正式策略需要：

- 按 CPU 架构和 kernel 类型采样；
- 比较大 grid 下 inline 阻塞 client 是否损害后续并行；
- 提供测试用环境变量强制 `dispatcher`/`inline`，不作为长期公共 API；
- 证据不足时默认关闭 inline。

不能用 `cube_dim.num_elems()` 单独代表元素数，也不能把 Apple Silicon 上的阈值复制到 x86 或服务器 CPU。

### 4.4 Inline 执行语义

合格任务执行：

```text
run_kernel(pliron_data)
atomic_counter.fetch_add(1, Release)
next_counter_step += 1
drop(keepalive)                    // run_kernel 已返回
```

要求：

- 与正常 `ComputeTask::compute` 使用同一个函数，避免两套 counter 逻辑；
- keepalive 在执行完成前一直存在；
- Host 的 `next_counter_step` 增量仍为 `cube_dim.num_elems()`，本阶段恰好为 1；
- 不经过 Threadpool lock、mpsc send 或 Worker 唤醒；
- `submit()`、read、sync、failure/taint 的外层 `ExecuteScope` 不变；
- 不在执行失败后对同一更新再跑 dispatcher，防止重复计算。

`run_kernel` 是已编译机器码入口，不属于普通可恢复 Rust 错误。上游评审需确认 panic/trap 行为是否允许与 Worker 路径不同；若必须保留 Worker panic 的特定 failure 行为，则 inline 暂停，而不能默默吞掉 panic。

### 4.5 不为首版追求“完全无线程池”

`CpuStream::new` 当前会通过 `Threadpool::get()` 初始化全局池。P1 只消除合格 launch 的每次锁/channel，不承诺消除线程池启动。

若 profile 证明 idle Worker 初始化或虚拟内存栈也是问题，可后续把 Threadpool 改为首次 parallel dispatch 时 lazy init。该改动需要单独验证：

- 全部 inline 的进程是否仍需要 Worker；
- 第一个非 inline launch 的冷启动；
- `needs_parallelism` 的 Worker 扩容；
- 多 stream 同时首次访问全局 OnceLock。

## 5. P2：连续任务的批投递

### 5.1 先找到真实批边界，不虚构 `Vec<ComputeTask>`

当前 Burn 每个 launch 独立进入 CPU server；简单在 `Threadpool` 增加 `execute_batch`，上层却没有 task list，不会自动产生收益。

可行的批边界位于 `SchedulerMultiStream` 把已排队任务交给实际执行 stream 时：

1. 按现有 Sequential/Interleave 策略生成实际入队顺序；
2. 按 Write、shared-memory Execute 和普通 Execute 分段；
3. 只对**普通 Execute 的连续区间**调用 `enqueue_tasks`；
4. Write/shared-memory 仍按原要求先 submit，不能跨 barrier 合并；
5. 每段内部保持原顺序和跨 stream 对齐。

这里必须按**实际接收任务的 `B::Stream` 执行队列**分段，而不是按 `ScheduleTask.stream_id` 重新分组。当前 Interleave 实现会先 flush 其他 schedule，再把多个逻辑 stream 的 task 按 round-robin 顺序交给第一个实际执行 stream；批投递不得为了凑同 logical-stream 的大批而改变这个顺序。`requires_isolation` 和 Sequential 路径则继续保持各 stream 独立。

可以先给上游 stream trait 增加逐项执行的默认 `enqueue_tasks`，CPU override 才真正合锁；其他后端行为不变。

### 5.2 一次锁的正确实现

`CpuStream::enqueue_tasks`：

1. 为区间中每个 launch 完成 `PreparedLaunch::prepare`；
2. 在**实际接收这段任务的 `CpuStream`** 上，以它当前的 `next_counter_step` 为起点计算每个 unit 的 readiness；不得按 `ScheduleTask.stream_id` 选择另一套 counter；
3. 只获取一次 `threadpool.lock()`；
4. 依次调用现有 send 逻辑；
5. Host 一次或按 task 更新同一个实际执行 stream 的 `next_counter_step`，保证与 Worker 的增量完全一致。

要求：

- 每个 task 自带 readiness/counter，不因为同批就跳过；
- 资源 keepalive 按 launch 隔离，不能把不同生命周期的资源混成一个无所有者 Vec；
- barrier task 继续 `ensure_workers` 和固定 Worker 映射；
- 普通 task 的 least-loaded 选择在整批持锁期间保持原子一致；
- channel send 次数不因批投递消失；
- Interleave 下两个非空 schedule 的 round-robin 顺序必须原样保留；若普通 Execute 连续段只有 1 项，就保持现有逐项路径，不得先按逻辑 stream 分组再制造“大批”；
- 长批不能无界阻塞其他 stream，需要设置批长上限并测量公平性。

### 5.3 验收重点

- 批长为 1 时与旧路径等价；
- 区间中包含 Write/shared memory 时正确拆段；
- 同批多 launch 的 counter 不多加、不少加；
- 两个 stream 同时提交时不出现长尾或饿死；
- 报告实际平均批长和锁次数，不用理论 99% 代替。

如果 AdamW Fused 后已经没有长连续批，P2 没有存在必要，应停止。

## 6. P3：Worker 退避只做工整实验

当前已有“200µs yield polling 后 blocking recv”，不是简单的无限 `yield_now` 循环。可比较：

- 当前策略；
- 先有限 `spin_loop`，再 `yield_now`，再 blocking recv；
- 缩短/延长 `IDLE_POLL`；
- 立即 blocking recv。

每种都测：

1. 微任务 p50/p95/p99；
2. 多 stream 尾延迟；
3. barrier 最后一个 unit 的唤醒延迟；
4. client 与 Worker 共用逻辑 CPU 时的饿死风险；
5. idle CPU、上下文切换和功耗代理指标。

不建议直接再引入一个条件变量通道：`mpsc::recv` 已提供阻塞等待，新增第二套 wakeup 状态可能造成丢失唤醒或重复通知。只有 profile 表明现有阻塞机制不适合时，才设计统一替代。

## 7. 内存回收：先纠正模型，再决定是否优化

### 7.1 不采用全局 thread-local free-list

以下设计不能直接实施：

```text
任意 Worker drop buffer
    → 放入当前线程 free-list
    → 下一次任意 stream/任意 kernel 直接复用
```

问题：

- task 可能在其他 stream 上仍有读写依赖；
- `ManagedResource` 与 pool/page/stream 所有权相关；
- CPU 指针可能被其他 unit 或 readback 使用；
- free-list 在线程退出时的回收和跨 CPU cache locality 未定义；
- 会绕过 failure/taint 和 relocation 的现有约束。

### 7.2 先定位 drop 成本归属

P0 需要区分：

- `keepalive` Vec 和 Box 析构；
- ManagedMemory handle/binding 引用计数；
- pool slice 合并；
- page deallocation；
- client `submit()` 等 Worker 完成；
- 真正的 storage allocator free。

若只是中间 tensor 数量过多，优先合并 AdamW kernel和复用输出，通常比改造全局内存系统更小、更安全。

### 7.3 有证据后才考虑 per-stream 延迟回收

若 profile 确认回收是主因，候选设计必须满足：

1. 回收请求携带原 stream/pool/allocation identity；
2. 以每个相关 stream 的 `required_counter` 集合或等价 fences 标记所有最后使用者完成的位置；
3. 只在 counter/fence 到达后复用；
4. 由原 owner 在明确 quiescent point 处理，或把 `MemoryManagement` 的所有权同步正式迁移；
5. readback、shared-memory、relocation、graph capture 和 failure/taint 都有对应规则；
6. free-list 有字节上限，避免把释放峰值变成永久驻留。

如果只是把回收延迟到下一个 `submit()`，要承认这改变的是成本出现的位置，不一定降低总成本。没有端到端收益就不做。

## 8. 正确性验证矩阵

### 8.1 Inline 必须覆盖

| 类别 | 用例 |
|---|---|
| 普通单 unit | 输出与 dispatcher 一致，counter 恰好增加 1 |
| 同流顺序 | A 写 buffer、B 读 buffer；A 未完成时 B 不得 inline 越过 |
| 跨 stream | S0 写、S1 读；对齐未完成时禁止 inline，完成后结果一致 |
| barrier | `needs_parallelism=true` 始终走 Worker，cube 多 unit 不死锁 |
| shared memory | 连续 shared launch 不复用仍被读取的块 |
| keepalive | 排队/执行/readback 期间资源有效，完成后才 drop |
| dynamic count | Host 解析 cube count 后才决定路径，count 0/1/大值均覆盖 |
| 大 grid | 工作量超阈值走 dispatcher，不长时间阻塞 client |
| 多 stream | 多线程同时 inline/dispatch，counter 和 pool 不串流 |
| 模式 | CompileOnly/Discard 不执行；普通 Execute 不受影响 |
| panic/trap | 按上游确定的错误契约验证，不能重复 launch |

### 8.2 批投递额外覆盖

- 普通 Execute 连续段长度 1、2、上限、超过上限；
- Execute→Write→Execute、Execute→shared→Execute 正确拆段；
- 同一批内 barrier 与普通 task 混合；
- Worker ready/aside 队列不丢 task；
- Sequential 下两个 stream 的 counter 各自独立；Interleave 下用两个**非空** schedule 验证 round-robin 顺序不变，且所有 readiness 都挂在实际接收任务的 `CpuStream` counter 上；
- 长批下其他 stream 仍能按调度策略执行。

### 8.3 内存改动额外覆盖

- readback 与回收竞争；
- 多 unit 共享一个 `SharedData`；
- relocation 与 free-list 同时发生；
- failure/taint 在延迟回收后仍指向正确 allocation；
- graph capture/replay；
- 反复分配达到 free-list 上限后的内存高水位。

## 9. 验证命令与上游集成

CPU 调度代码位于 CubeCL 仓库，而不是 Burn `crates/burn-cpu` 内。实施分两步：

```sh
# 在 CubeCL 的指定工作树和 revision 上
cargo test -p cubecl-server
cargo test -p cubecl-cpu

# 在 Burn 中更新到包含改动的明确 CubeCL rev 后
# 当前 xtask 没有 cpu backend alias；仓库级检查使用默认 Flex，CubeCL CPU 由下面的直接测试覆盖
cargo run-checks
cargo test -p burn-optim --no-default-features --features std,cpu
cargo test -p burn-optim --no-default-features --features std,cpu,adamw-bench-observability
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu,adamw-bench-observability -- --verify-path
cargo bench -p burn-optim --bench adamw --no-default-features --features std,cpu
```

要求：

- PR 描述写清 CubeCL 上游 commit 和 Burn rev 更新关系；
- Burn `Cargo.lock` 只允许目标 CubeCL 源/rev 及其必要依赖变化；
- 上游测试通过后仍需 Burn 端到端验证；
- Burn 的 raw AdamW 对照必须使用上述 `--no-default-features --features std,cpu`，并确保 `burn-nn` dev-dependency 也只启用 `std`，防止默认 feature 链重新启用 Fusion；测试断言 `!burn_cubecl::FUSED`，所有会触发专用路径的测试共用串行锁并在锁内取计数差值。计数器位于实际提交 launch/chunk 的 `burn-cubecl`，其 `adamw-bench-observability` feature 由 `burn-optim` 通过 `burn-cubecl?/adamw-bench-observability` 转发；依赖上层 `cfg(test)` 不会打开依赖 crate 的观测代码，所以路径计数测试必须显式启用该 feature。独立 benchmark 先通过非默认 feature 的 `#[doc(hidden)] pub` 接口验证实际 launch/chunk 数，再以未启用该 feature 的同 raw 构建计时；bench 的 verification import、helper 和分派函数也全部用该 feature 条件编译，无 feature 构建遇到 `--verify-path` 只提示重建并失败退出，不引用被移除的接口。不能只比较输出，也不能把观测开销计入性能结果；
- `burn-optim` 的 `adamw` bench target 必须在 `Cargo.toml` 显式设置 `[[bench]] name = "adamw", path = "benches/adamw.rs", harness = false`，计时入口调用 `divan::main()`；`--verify-path` 在 Divan 前解析，验证入口断言预期 case 全部实际完成且数量非零，无观测 feature 时只提示重建并失败退出，不能由 libtest harness 或 0-case 运行产生虚假的性能/路径结论；
- `cubecl-server` 负责 scheduler trait 和 Sequential/Interleave 单元测试；`cubecl-cpu` 还需提供经 `CpuServer` 驱动的跨 crate 集成测试，覆盖实际执行 stream 的分段、counter/readiness、Write/shared 边界、批长 1，以及两个非空 schedule 的 Interleave 顺序和公平性；
- 无 CUDA/ROCm/Metal 硬件时，CPU runtime 改动不得被宣传为已在这些 runtime 验证；
- 若只完成 profiling 或 P1，不把 P2/P3 写成已验证。

## 10. 阶段门禁与 PR 拆分

### PR 0：仅 instrumentation 和 benchmark

不改变调度结果。先提交测量方法、原始数据和可复现 workload，再确定是否继续。

### PR 1：PreparedLaunch 机械重构

只把准备阶段与 `&mut Threadpool` 调度分离，所有 launch 仍走旧路径。证明 counter、shared memory 和 keepalive 行为未变。机械重构与行为改变分开，便于审查。

### PR 2：受限 Inline

加入完整准入条件、强制路径测试配置、并发回归和 benchmark。若只在 microbenchmark 有收益而 AdamW/多 stream 回退，则不默认开启。

### PR 3：连续 Execute 批投递（有数据才做）

先实现默认逐项的 `enqueue_tasks`，再让 CPU override 合锁。提交实际批长、锁次数和公平性数据。

### PR 4：退避或回收（分别独立）

polling 与 memory 不捆 PR；每个都依赖 P0 中对应瓶颈的证据。无证据就关闭该方向，不影响前面已验证的优化。

## 11. 与 AdamW 方案的组合验收

最终至少需要以下对照，而不是只比较“旧 CPU vs 新 CPU”：

```text
Reference AdamW + dispatcher
Fused AdamW + dispatcher
Reference AdamW + inline（若合格）
Fused AdamW + inline（若合格）
Flex Reference AdamW
```

分析原则：

- `Fused + dispatcher` 与 `Reference + dispatcher` 的差值主要反映上层融合；
- `Fused + inline` 与 `Fused + dispatcher` 的差值才反映 CPU 调度改动；
- Flex 只作另一种实现的对照，不把全部差值归结为某一个锁；
- batch size 只作为端到端训练对照，纯 optimizer benchmark 使用固定参数/梯度 shape；
- 收益按 workload 报告，不外推为所有模型和所有 CPU 的固定倍数。

## 12. 锁定源码索引

以下来源均对应 `d88b0febb50e7b83a8548125f6a701c12f11a2ba`：

- [`threadpool/mod.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-cpu/src/compute/threadpool/mod.rs)：全局锁、task 构造、barrier Worker 和 keepalive。
- [`scheduler/dispatcher.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-cpu/src/compute/threadpool/scheduler/dispatcher.rs)：least-loaded、固定 barrier Worker、IDLE_POLL 和阻塞 recv。
- [`compute_task.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-cpu/src/compute/threadpool/compute_task.rs)：ready 与 counter 增量。
- [`stream.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-cpu/src/compute/stream.rs)：enqueue、submit、shared-memory drain 和 per-stream memory pools。
- [`server.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-cpu/src/compute/server.rs)：dynamic count、跨 stream 对齐、ExecuteScope 和 failure 路径。
- [`stream/scheduler.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-server/src/stream/scheduler.rs)：任务队列、Sequential/Interleave 的真实入队顺序和 isolation。
- [`PlironEngine::run_kernel`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-llvm/src/cpu/jit/engine.rs)：编译后函数入口。
- [`memory_manage.rs`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-server/src/memory_management/memory_manage.rs)：MemoryManagement 所有权和 cleanup。
- [`ManagedMemoryHandle`](https://github.com/tracel-ai/cubecl/blob/d88b0febb50e7b83a8548125f6a701c12f11a2ba/crates/cubecl-runtime/src/memory_management/handle.rs)：handle/binding 生命周期和 can_mut。

这些链接用于约束设计；任何后续实现仍需以实际选定的上游 revision 重新核对。
