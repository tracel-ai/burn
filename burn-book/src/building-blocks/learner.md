# Learner

The [burn-train](https://github.com/tracel-ai/burn/tree/main/crates/burn-train) crate encapsulates
multiple utilities for training deep learning models. The goal of the crate is to provide users with
a well-crafted and flexible training loop, so that projects do not have to write such components
from the ground up. Most of the interactions with `burn-train` will be with the `SupervisedTraining`
struct, briefly presented in the previous [training section](../basic-workflow/training.md). This
struct enables you to configure the training loop, offering support for registering metrics,
enabling logging, checkpointing states, using multiple devices, and so on.

There are still some assumptions in the current provided APIs, which may make them inappropriate for
your learning requirements. Indeed, they assume your model will learn from a training dataset and be
validated against another dataset. This is the most common paradigm, allowing users to do both
supervised and unsupervised learning as well as fine-tuning. However, for more complex requirements,
creating a [custom training loop](../custom-training-loop.md) might be what you need.

## Usage

The `SupervisedLearning` struct must be created with the training and validation dataloaders. It
provides you with numerous options when it comes to configurations.

| Configuration          | Description                                                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Training Metric        | Register a training metric                                                                                                              |
| Validation Metric      | Register a validation metric                                                                                                            |
| Training Metric Plot   | Register a training metric with plotting (requires the metric to be numeric)                                                            |
| Validation Metric Plot | Register a validation metric with plotting (requires the metric to be numeric)                                                          |
| Metric Logger          | Configure the metric loggers (default is saving them to files)                                                                          |
| Renderer               | Configure how to render metrics (default is CLI)                                                                                        |
| Grad Accumulation      | Configure the number of steps before applying gradients                                                                                 |
| File Checkpointer      | Configure how the model, optimizer and scheduler states are saved                                                                       |
| Num Epochs             | Set the number of epochs                                                                                                                |
| Devices                | Set the devices to be used                                                                                                              |
| Checkpoint             | Restart training from a checkpoint                                                                                                      |
| Application logging    | Configure the application logging installer (default is writing to `experiment.log`)                                                    |
| Training Strategy      | Use a custom training strategy, allowing you to use your own training loop with all the capabilities of the `SupervisedTraining` struct |

When the training is configured to your liking, you can then move forward to running the training.
The `launch` method requires a learner object providing: the model, the optimizer and the learning
rate scheduler. Note that the latter can be a simple float if you want it to be constant during
training. See the [learning rate scheduler section](./lr-scheduler.md) for the available schedulers.

The `launch` method will start the training and return the trained model once finished.

Again, please refer to the [training section](../basic-workflow/training.md) for a relevant code
snippet.

## Parameter Groups

It's common to use different learning rates or optimizer settings for different parts of a model.
Burn's `ParamGroup` routes module parameters by path or ID. Optimizers and learning-rate schedulers
use the same matching rules but can be configured independently.

```rust,ignore
let lr_scheduler_base = ComposedLrSchedulerConfig::new()
    .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 2000))
    .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2000))
    .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10000));
let lr_scheduler = lr_scheduler_base.init().unwrap().with_group(
    ParamGroup::from_predicate("conv"),
    LinearLrSchedulerConfig::new(1e-6, 1e-3, 14000)
        .build()
        .unwrap(),
);

let optim = AdamWConfig::new()
    .with_cautious_weight_decay(true)
    .with_weight_decay(5e-5)
    .init();

let result = training.launch(Learner::new(
    model,
    optim,
    lr_scheduler,
));
```

For group-specific optimizers, matching precedence, gradient clipping, and optimizer state, see
[Optimizer](./optimizer.md#parameter-groups).

## Artifacts

When creating a `SupervisedTraining` instance, all the collected data will be saved under the
directory provided as the argument to the `new` method. Here is an example of the data layout for a
model checkpointed to the burnpack format, with the accuracy and loss metrics registered:

```
├── experiment.log
├── checkpoint
│   ├── model-1.bpk
│   ├── optim-1.bpk
│   └── scheduler-1.bpk
│   ├── model-2.bpk
│   ├── optim-2.bpk
│   └── scheduler-2.bpk
├── train
│   ├── epoch-1
│   │   ├── Accuracy.log
│   │   └── Loss.log
│   └── epoch-2
│       ├── Accuracy.log
│       └── Loss.log
└── valid
    ├── epoch-1
    │   ├── Accuracy.log
    │   └── Loss.log
    └── epoch-2
        ├── Accuracy.log
        └── Loss.log
```

You can choose to save or synchronize that local directory with a remote file system, if desired.
The file checkpointer is capable of automatically deleting old checkpoints according to a specified
configuration.
