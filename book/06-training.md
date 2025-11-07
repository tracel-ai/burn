# Chapter 6: Training Models with Burn

Having a model is only half the battle; you also need to train it. This is where the `burn-train` crate comes in. It provides a flexible and powerful training infrastructure centered around the `Learner` struct, which is configured and created using the `LearnerBuilder`.

## The `LearnerBuilder`: Composing a Training Process

Instead of creating a `Learner` directly, you use the `LearnerBuilder` (found in `crates/burn-train/src/learner/builder.rs`). This is a classic example of the **Builder pattern**, which is used throughout Burn to handle complex object creation. It allows you to configure the training process step-by-step in a clear and readable way.

A typical training setup looks like this:

```rust
use burn::train::{LearnerBuilder, ClassificationOutput};
use burn::prelude::*;

// Assume Model, Optimizer, LrScheduler, and Dataloaders are defined

fn run_training<B: AutodiffBackend>(device: &B::Device) {
    let learner = LearnerBuilder::new("/tmp/my-model-checkpoint")
        .with_batch_size(32)
        .num_epochs(10)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .build(
            model,
            optimizer,
            lr_scheduler,
            train_dataloader,
            valid_dataloader,
        );

    let model_trained = learner.fit();
}
```

This snippet showcases the fluent API of the `LearnerBuilder`. Let's break down the key components.

### Core Components of a Learner

The `Learner` struct itself (defined in `crates/burn-train/src/learner/base.rs`) is a container for all the pieces needed for a training loop:

```rust
// A simplified view of the Learner struct
pub struct Learner<LC: LearnerComponentTypes> {
    pub(crate) model: LC::Model,
    pub(crate) optim: LC::Optimizer,
    pub(crate) lr_scheduler: LC::LrScheduler,
    pub(crate) num_epochs: usize,
    // ... and many other fields for checkpointing, logging, etc.
}
```

*   **Model (`model`)**: This is the neural network you want to train. It must be a struct that implements `Module` and `TrainStep`.
*   **Optimizer (`optim`)**: The optimizer is responsible for updating the model's parameters based on the computed gradients. Burn provides common optimizers like Adam and SGD in the `burn-optim` crate.
*   **Learning Rate Scheduler (`lr_scheduler`)**: This component adjusts the learning rate during training, which can significantly improve convergence.
*   **DataLoaders**: The `build` method takes a training and a validation dataloader. These are responsible for loading data in batches from your dataset. The `burn-dataset` crate provides tools for creating custom datasets.

### The `TrainStep` and `ValidStep` Traits

For a model to be trainable by the `Learner`, it must implement two important traits: `TrainStep` and `ValidStep`. These traits define the logic for a single step of training and validation, respectively.

```rust
use burn::train::{TrainStep, ValidStep, TrainOutput, ClassificationOutput};
use burn::prelude::*;

// Assuming a `MyModel` struct and a `MyBatch` type are defined.

impl<B: AutodiffBackend> TrainStep<MyBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(batch); // Perform the forward pass
        let grads = item.loss.backward(); // Compute gradients from the loss

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<MyBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyBatch<B>) -> ClassificationOutput<B> {
        self.forward(batch) // Just perform the forward pass
    }
}
```
*   **`TrainStep::step`**: This function defines what happens for a single training batch. It takes the model and a batch of data, performs the forward pass to get the output (which includes the loss), computes the gradients using `.backward()`, and returns a `TrainOutput` containing the model, the gradients, and the output. The `Learner` then uses these gradients to update the model's parameters via the optimizer.
*   **`ValidStep::step`**: This function is simpler. It's only responsible for performing the forward pass on a validation batch and returning the output. No gradients are computed, as we don't update the model during validation.

This separation of concerns allows you to define the core logic of your model's training and validation steps, while the `Learner` handles the boilerplate of the training loop itself (iterating over epochs and batches, calling the optimizer, logging metrics, etc.).

## The Training Process: `.fit()`

Once the learner is built, you start the training by calling the `.fit()` method. This method takes ownership of the learner and executes the entire training loop:

1.  **Epoch Loop**: It iterates for the specified number of epochs.
2.  **Training Step**: In each epoch, it iterates through the training dataloader, batch by batch. For each batch, it calls the model's `TrainStep::step` method and then uses the returned gradients to update the model weights with the optimizer.
3.  **Validation Step**: After each training epoch, it iterates through the validation dataloader, calling the model's `ValidStep::step` method for each batch and calculating the validation metrics.
4.  **Checkpointing and Logging**: Throughout the process, it saves model checkpoints and logs metrics according to the configuration you provided to the `LearnerBuilder`.

This structured approach, combining a powerful `Learner` with a flexible `LearnerBuilder`, provides a robust and extensible system for training a wide variety of models in Burn.

---

## Exercises

1.  **Explore Optimizers**: Look at the `burn-optim` crate. Find the implementation for the `Adam` optimizer. What are some of its configuration parameters (e.g., `beta1`, `beta2`, `epsilon`)?
2.  **Custom Metric**:
    a.  Implement a new metric called `MeanAbsoluteError`. It should implement the `Metric` trait. The `update` method will receive a `ClassificationOutput` and should calculate the mean absolute error between the `output` and `targets`.
    b.  Add an instance of your new metric to the `LearnerBuilder` for both training and validation.
3.  **Thought Experiment**: Why is it important for the `TrainStep` trait to be defined on an `AutodiffBackend`, while the `ValidStep` trait only requires a standard `Backend`?
