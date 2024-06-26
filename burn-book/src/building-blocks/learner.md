# Learner

The [burn-train](https://github.com/tracel-ai/burn/tree/main/crates/burn-train) crate encapsulates
multiple utilities for training deep learning models. The goal of the crate is to provide users with
a well-crafted and flexible training loop, so that projects do not have to write such components
from the ground up. Most of the interactions with `burn-train` will be with the `LearnerBuilder`
struct, briefly presented in the previous [training section](../basic-workflow/training.md). This
struct enables you to configure the training loop, offering support for registering metrics,
enabling logging, checkpointing states, using multiple devices, and so on.

There are still some assumptions in the current provided APIs, which may make them inappropriate for
your learning requirements. Indeed, they assume your model will learn from a training dataset and be
validated against another dataset. This is the most common paradigm, allowing users to do both
supervised and unsupervised learning as well as fine-tuning. However, for more complex requirements,
creating a [custom training loop](../custom-training-loop.md) might be what you need.

## Usage

The learner builder provides numerous options when it comes to configurations.

| Configuration          | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| Training Metric        | Register a training metric                                                     |
| Validation Metric      | Register a validation metric                                                   |
| Training Metric Plot   | Register a training metric with plotting (requires the metric to be numeric)   |
| Validation Metric Plot | Register a validation metric with plotting (requires the metric to be numeric) |
| Metric Logger          | Configure the metric loggers (default is saving them to files)                 |
| Renderer               | Configure how to render metrics (default is CLI)                               |
| Grad Accumulation      | Configure the number of steps before applying gradients                        |
| File Checkpointer      | Configure how the model, optimizer and scheduler states are saved              |
| Num Epochs             | Set the number of epochs.                                                      |
| Devices                | Set the devices to be used                                                     |
| Checkpoint             | Restart training from a checkpoint                                             |
| Application logging    | Configure the application logging installer (default is writing to `experiment.log`)                                   |

When the builder is configured at your liking, you can then move forward to build the learner. The
build method requires three inputs: the model, the optimizer and the learning rate scheduler. Note
that the latter can be a simple float if you want it to be constant during training.

The result will be a newly created Learner struct, which has only one method, the `fit` function
which must be called with the training and validation dataloaders. This will start the training and
return the trained model once finished.

Again, please refer to the [training section](../basic-workflow/training.md) for a relevant code
snippet.

## Artifacts

When creating a new builder, all the collected data will be saved under the directory provided as
the argument to the `new` method. Here is an example of the data layout for a model recorded using
the compressed message pack format, with the accuracy and loss metrics registered:

```
├── experiment.log
├── checkpoint
│   ├── model-1.mpk.gz
│   ├── optim-1.mpk.gz
│   └── scheduler-1.mpk.gz
│   ├── model-2.mpk.gz
│   ├── optim-2.mpk.gz
│   └── scheduler-2.mpk.gz
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
