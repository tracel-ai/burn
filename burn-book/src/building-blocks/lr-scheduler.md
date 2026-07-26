# Learning Rate Scheduler

Learning rate schedulers control how the learning rate evolves during training. A scheduler is
built from its configuration and passed to the learner along with the model and the optimizer, as
shown in the [learner section](./learner.md). A constant learning rate can be provided as a simple
float. We currently offer the following schedulers.

| Scheduler        | Description                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| Constant         | Keep the learning rate fixed during training                                       |
| Linear           | Interpolate linearly between an initial and final learning rate                    |
| Cosine Annealing | Anneal the learning rate following a cosine curve                                  |
| Exponential      | Multiply the learning rate by a constant factor at every step                      |
| Noam             | Warm up linearly, then decay proportionally to the inverse square root of the step |
| Step             | Multiply the learning rate by a constant factor at fixed intervals                 |
| Composed         | Combine schedulers per parameter group (see the [learner section](./learner.md#multiple-optimizers)) |
| Sequential       | Run different schedulers during non-overlapping parts of training                  |

## Sequential learning rate schedules

Use `SequentialLrSchedulerConfig` when different schedulers should run during non-overlapping parts
of training. Milestones count scheduler steps: in this warmup example, the linear scheduler produces
the first 1,000 learning rates and the cosine scheduler takes over on step 1,000.

```rust,ignore
let lr_scheduler = SequentialLrSchedulerConfig::new(
    vec![
        LinearLrSchedulerConfig::new(1e-6, 1e-3, 1_000).into(),
        CosineAnnealingLrSchedulerConfig::new(1e-3, 9_000).into(),
    ],
    vec![1_000],
)
.init()?;
```
