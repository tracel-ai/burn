# Learner

The [burn-train](https://github.com/burn-rs/burn/tree/main/burn-train) crate encapsulate multiple utilities when it comes to training deep learning mdoels.
The goal of the crate is to provide users with a well crafted and flexible training loop, so that every project don't have to rewrite that stuff from scratch.
Most of the interactions with `burn-train` will be with the `LearnerBuilder` struct, birefly presented in the [previous section](../basic-workflow/training.md).
This struct let you configure the training loop with support for registering metrics, enabling logging, checkpointing states, using multile devices and so on.

There are still some assumptions in the current provided APIs that may make them inappropriate for your learning requirements.
First, they assume your model will learn from a training dataset and be validated against another dataset.
This is the most common paradigm and allows users to do supervised and unsupervised learning as well as fine-tuning.
However, if you have more complex requirements, creating a [custom training loop] (./advanced/custom-training-loop.md) might be what you need.

## Usage

The learner builder provides a lot of options when it comes to configurations.

| Configurations         | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| Training Metric        | Register a training metric                                                    |
| Validation Metric      | Register a validation metric                                                  |
| Training Metric Plot   | Register a training metric with plotting (require the metric to be numeric)   |
| Validation Metric Plot | Register a validation metric with plotting (require the metric to be numeric) |
| Metric Logger          | Configure the metric loggers (default is saving them to files)                |
| Renderer               | Configure how to render metrics (default is CLI)                              |
| Grad Accumulation      | Configure the number of steps before applying gradients                       |
| File Checkpointer      | Configure how the model, optimizer and scheduler state are saved              |
| Num Epochs             | Set the number of epochs.                                                     |
| Devices                | Set the devices to be used                                                    |
| Checkpoint             | Restart training from a checkpoint                                            |

When the builder is condiured as your liking, we can them move forward to build the learner.
The build method requires three input: the model, the optimizer and the learning rate scheduler.
Note that the learning rate scheduler can be a simple float if you want it to be constant during training.

The result will be a newly created learner struct, which has only one method.
You can call the `fit` function with the provided training and validation dataloaders.
This will start the training, and once finished, will return the trained model.

## Artifacts

When creating a new builder, all the collected data will be saved under the directory provided in the `new` methods.
Here's an example of the saved data layour with a model saved using the compressed message pack format, with the accuracy and loss metrics registered:

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

You could save or sync that local directory with a remove file system if you want.
Note that the file checkpointer can delete automatically olds checkpoints based on a configuration.
