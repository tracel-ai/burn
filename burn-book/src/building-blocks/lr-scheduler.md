# Learning Rate Scheduler

Learning rate schedulers control how the learning rate evolves during training. A scheduler is built
from its configuration and passed to the learner along with the model and the optimizer, as shown in
the [learner section](./learner.md). A constant learning rate can be provided as a simple float. We
currently offer the following schedulers.

| Scheduler        | Description                                                                        |
| ---------------- | ---------------------------------------------------------------------------------- |
| Constant         | Keep the learning rate fixed during training                                       |
| Linear           | Interpolate linearly between an initial and final learning rate                    |
| Cosine Annealing | Follow a cosine curve without warm restarts                                        |
| Exponential      | Multiply the learning rate by a constant factor at every step                      |
| Noam             | Warm up linearly, then decay proportionally to the inverse square root of the step |
| Step             | Multiply the learning rate by a constant factor at fixed intervals                 |
| Composed         | Combine simultaneously advancing schedules by multiplication, sum, or average      |
| Sequential       | Run different schedulers during non-overlapping parts of training                  |

`ComposedLrSchedulerConfig` advances all its component schedules at every step and combines their
values. It multiplies them by default; `with_reduction(SchedulerReduction::Sum)` or
`with_reduction(SchedulerReduction::Avg)` changes the reduction. Use `SequentialLrSchedulerConfig`
for successive phases, such as warmup followed by decay.

Parameter groups are independent of composition and sequencing. `ModuleLrSchedulerConfig` assigns
different schedules to different groups of parameters, as shown in the
[learner section](./learner.md#parameter-groups). Scheduler configuration `init()` methods return a
`ModuleLrScheduler`; in a custom loop, its `step()` returns a `ModuleLearningRate` that can be passed
directly to the optimizer.

## Cosine annealing

`CosineAnnealingLrSchedulerConfig::new(initial_lr, num_iters)` starts at `initial_lr` on the first
call to `step()`. After `num_iters` further steps, it reaches `min_lr` (zero by default). If you keep
stepping, the learning rate rises again along the cosine curve; it neither stays at the minimum nor
resets abruptly to the initial value.

In Burn 0.21, this scheduler restarted after reaching the minimum. Burn 0.22 removes that reset, so
the same configuration produces different learning rates after the first descent. If your training
relies on warm restarts, implement the resets explicitly in a custom scheduler or training loop.

## Sequential learning rate schedules

Use `SequentialLrSchedulerConfig` when different schedulers should run during non-overlapping parts
of training. Milestones count scheduler steps: in this warmup example, the linear scheduler produces
the first 1,000 learning rates and the cosine scheduler takes over on step 1,000.

```rust,ignore
use burn::optim::lr_scheduler::{
    cosine::CosineAnnealingLrSchedulerConfig,
    linear::LinearLrSchedulerConfig,
    sequential::SequentialLrSchedulerConfig,
};

let lr_scheduler = SequentialLrSchedulerConfig::new(
    vec![
        LinearLrSchedulerConfig::new(1e-6, 1e-3, 1_000).into(),
        CosineAnnealingLrSchedulerConfig::new(1e-3, 9_000).into(),
    ],
    vec![1_000],
)
.init()?;
```
