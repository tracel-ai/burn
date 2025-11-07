# Chapter 6: Training Models with Burn

Having a model is only half the battle; you also need to train it. This is where the `burn-train` crate comes in. It provides a flexible and powerful training infrastructure centered around the `Learner` struct, which is configured and created using the `LearnerBuilder`.

## The `LearnerBuilder`: Composing a Training Process

Instead of creating a `Learner` directly, you use the `LearnerBuilder` (found in `crates/burn-train/src/learner/builder.rs`). This is a classic example of the **Builder pattern**, which is used throughout Burn to handle complex object creation. It allows you to configure the training process step-by-step in a clear and readable way.

### Data Flow Diagram of `learner.fit()`

This diagram illustrates the entire process that the `learner.fit()` method orchestrates.

```
+------------------+     +-------------------+
| Train DataLoader | --> |  Validation Data  |
+------------------+     +-------------------+
        |                        |
        V                        V
.-------'------------------------'-------.
|             `learner.fit()`            |
|                                        |
|   For each Epoch:                      |
|   +--------------------------------+   |
|   |    For each Training Batch:    |   |
|   |  1. Get Batch from Dataloader  |   |
|   |  2. `model.step()` (forward)   |   |
|   |  3. `loss.backward()`          | ----> Computation Graph
|   |  4. `optimizer.step()`         |   |
|   +--------------------------------+   |
|                                        |
|   +--------------------------------+   |
|   |   For each Validation Batch:   |   |
|   |  1. Get Batch from Dataloader  |   |
|   |  2. `model.step()` (forward)   |   |
|   |  3. Calculate Metrics          |   |
|   +--------------------------------+   |
|                                        |
| Checkpointing & Logging throughout...  |
`----------------------------------------'
```

## Runnable Example: A Complete Training Loop

This example demonstrates a complete training loop for a simple model on dummy data.

```rust
use burn::prelude::*;
use burn::train::{LearnerBuilder, ClassificationOutput, TrainStep, ValidStep, TrainOutput};
use burn::nn::{Linear, LinearConfig, ReLU, loss::CrossEntropyLoss};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::FakeDataset;
use burn::optim::{Adam, AdamConfig};

// Use the model from the previous chapter
#[derive(Module, Debug)]
pub struct MyModel<B: Backend> {
    linear1: Linear<B>,
    relu: ReLU,
    linear2: Linear<B>,
}
#[derive(Config)]
pub struct MyModelConfig {
    d_input: usize,
    d_hidden: usize,
    d_output: usize,
}
impl MyModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModel<B> {
        MyModel {
            linear1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            relu: ReLU::new(),
            linear2: LinearConfig::new(self.d_hidden, self.d_output).init(device),
        }
    }
}

// Define the forward pass and the step functions
impl<B: Backend> MyModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.relu.forward(x);
        self.linear2.forward(x)
    }

    pub fn forward_classification(&self, item: MyBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss = CrossEntropyLoss::new(None, &output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MyBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch);
        let grads = item.loss.backward();
        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<MyBatch<B>, ClassificationOutput<B>> for MyModel<B> {
    fn step(&self, batch: MyBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}

// Define a simple batch struct for our dummy data
#[derive(Clone, Debug)]
pub struct MyBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

// Main training function
fn run_training_example() {
    type MyBackend = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;
    let device = Default::default();

    // Create dataloaders with fake data
    let train_loader = DataLoaderBuilder::new(FakeDataset::<MyBatch<MyBackend>>::new(200))
        .batch_size(32)
        .build(&device);
    let valid_loader = DataLoaderBuilder::new(FakeDataset::<MyBatch<MyBackend>>::new(50))
        .batch_size(32)
        .build(&device);

    // Create model, optimizer, and learner
    let config = MyModelConfig::new(10, 32, 5);
    let model = config.init::<MyBackend>(&device);
    let optim = AdamConfig::new().init();

    let learner = LearnerBuilder::new("/tmp/my-training-run")
        .num_epochs(3)
        .build(model, optim, 1e-4.into(), train_loader, valid_loader);

    // Run the training loop
    let model_trained = learner.fit();

    println!("Training complete!");
}
```

This separation of concerns allows you to define the core logic of your model's training and validation steps, while the `Learner` handles the boilerplate of the training loop itself (iterating over epochs and batches, calling the optimizer, logging metrics, etc.).

---

## Exercises

1.  **Explore Optimizers**: Look at the `burn-optim` crate. Find the implementation for the `Adam` optimizer. What are some of its configuration parameters (e.g., `beta1`, `beta2`, `epsilon`)?
2.  **Custom Metric**:
    a.  Implement a new metric called `MeanAbsoluteError`. It should implement the `Metric` trait. The `update` method will receive a `ClassificationOutput` and should calculate the mean absolute error between the `output` and `targets`.
    b.  Add an instance of your new metric to the `LearnerBuilder` for both training and validation.
3.  **Thought Experiment**: Why is it important for the `TrainStep` trait to be defined on an `AutodiffBackend`, while the `ValidStep` trait only requires a standard `Backend`?
4.  **Learning Rate Schedulers**: The `build` method in the example passes a simple float (`1e-4.into()`) as the learning rate scheduler. Look at the `burn-optim/src/lr_scheduler` directory. Find one of the more advanced schedulers (like `CosineAnnealingLR`) and explain what it does. How would you integrate it into the `LearnerBuilder`?
