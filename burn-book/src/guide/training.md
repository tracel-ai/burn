# Training

Now we are ready to write the necessary code to traing our model on the MNIST dataset.
The model should output an item that can be understood by the learner.
The output struct is used for all metrics calculated during the training.
So you normally include all the necessary information to calculate any metric that you want for a task.

Burn provides two basic output type: `ClassificationOutput` and `RegressionOutput` where they implement the necesary trait to be used with metrics.
It is possible to create your own item, but it is behoud the scope of this guide.

Since the MNIST task is a classification problem, we will use the `ClassificationOutput`


```rust, ignore
impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
```

As we can see in the previous codeblock, we use the cross entropy loss module without any padding token to calculate the loss.
We then return the classification output with the loss, the output tensor containing all logits and the targets.
The next part is to implement the training and validation step for our model.

```rust, ignore
impl<B: ADBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
```

Here we define the input and output type as generic argument in the `TrainStep` and `ValidStep` which are `MNISTBatch` and `ClassificationOutput`.
The training step requires the gradients to be calculated, we just need to call `backward()` on the loss to compute them.
Note that contairy to PyTorch, gradients arent store alongside each tensor parameters, but are returned by the backward pass `let gradients = loss.backward();`.
The gradient of a parameter can be obtained with the grad function `let grad = tensor.grad(&gradients);`, but it's not necessary when using the learning struct and the optimizers, it can be handly when debugging.
One of the difference between the training and the validation step is that the training step required the backend to implement `ADBackend` and not just `Backend`.
Otherwise, the `backward` functions isn't available since the backend doesn't support autodiff.
We will see later how to create a backend with autodiff support.

The next step is to create the actually training setup.

```rust, ignore
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: ADBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(&format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    // Data
    let batcher_train = MNISTBatcher::<B>::new(device.clone());
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::test());

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(1, CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .expect("Failed to save trained model");
}
```

It is a good practice to use the `Config` derive to create the experiment configuration.
The first thing we are doing is making sure the `artifact_dir` is created using the standard rust library.
All checkpoints, logging and metrics will be stored under the given directory.
We then initiazed our dataloaders using our batcher that we created previously.
Note that the backend used for the validation batcher is `B::InnerBackend`, since it doesn't need autodiff.
The autodiff capabilities are avaiable through a type system, so it's hard to forget to deactivate gradient calculation.
We then create our learning with the accuracy and loss metric on both training and validation steps along with the device and the epoch.
Note that we configure the checkpointer using the `CompactRecorder`.
Structs implementing the `Recorder` trait are capable of saving records for persistency.
We the  build the learning builder with the model, the optimizer and the learning rate.
Note that the third argument of the build function takes a learning rate scheduler, when providing with a float, it is automatically transformed into a constant learning rate scheduler.
The learning rate isn't part of the optimizer config as it is often done in other frameworks, but instead passed as parameter when executing the optimizer step.
This avoid having to mutate a state and is more functional.
You won't notice this when using the learner struct, but if you implement your own training loop, now you know.(advance / intermediate guide custom training loop)

When the learning is created, we can simply call fit and provide the tranining and validation dataloaders.
Note that we use the test set as the validation set in this example for simplicity, but we don't advise it for real usage 😉.
Finally, the trained model is returned by the fit method, and the only remaining part is saving the trained weights using the `CompactRecorder`.
