# Training

We are now ready to write the necessary code to train our model on the MNIST dataset.
Instead of a simple tensor, the model should output an item that can be understood by the learner, a struct whose responsibility is to apply an optimizer to the model.
The output struct is used for all metrics calculated during the training.
Therefore it should include all the necessary information to calculate any metric that you want for a task.

Burn provides two basic output types: `ClassificationOutput` and `RegressionOutput`. They implement the necessary trait to be used with metrics. It is possible to create your own item, but it is beyond the scope of this guide.

Since the MNIST task is a classification problem, we will use the `ClassificationOutput` type. 


```rust , ignore
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

As evident from the preceding code block, we employ the cross-entropy loss module for loss calculation, without the inclusion of any padding token.
We then return the classification output containing the loss, the output tensor with all logits and the targets.

Please take note that tensor operations receive owned tensors as input. For reusing a tensor multiple times, you need to use the clone function. There's no need to worry; this process won't involve actual copying of the tensor data. Instead, it will simply indicate that the tensor is employed in multiple instances, implying that certain operations won't be performed in place. In summary, our API has been designed with owned tensors to optimize performance.

Moving forward, we will proceed with the implementation of both the training and validation steps for our model.

```rust , ignore
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

Here we define the input and output types as generic arguments in the `TrainStep` and `ValidStep`. We will call them `MNISTBatch` and `ClassificationOutput`.
In the training step, the computation of gradients is straightforward, necessitating a simple invocation of `backward()` on the loss.
Note that contrary to PyTorch, gradients are not store alongside each tensor parameter, but are rather returned by the backward pass, as such: `let gradients = loss.backward();`.
The gradient of a parameter can be obtained with the grad function: `let grad = tensor.grad(&gradients);`. Although it is not necessary when using the learner struct and the optimizers, it can prove to be quite useful when debugging or writing custom training loops.
One of the differences between the training and the validation steps is that the former requires the backend to implement `ADBackend` and not just `Backend`.
Otherwise, the `backward` function is not available, as the backend does not support autodiff.
We will see later how to create a backend with autodiff support.

Let us move on to establishing the practical training configuration.

```rust , ignore
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
        .save(format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

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

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}
```

It is a good practice to use the `Config` derive to create the experiment configuration.
In the `train` function, the first thing we are doing is making sure the `artifact_dir` exists, using the standard rust library for file manipulation.
All checkpoints, logging and metrics will be stored under the this directory.
We then initialize our dataloaders using our previously created batcher.
Since no automatic differentiation is needed during the validation phase, the backend used for the corresponding batcher is `B::InnerBackend` (see [Backend](./backend.md)).
The autodiff capabilities are available through a type system, making it nearly impossible to forget to deactivate gradient calculation.

Next, we create our learner with the accuracy and loss metric on both training and validation steps along with the device and the epoch.
We also configure the checkpointer using the `CompactRecorder` to indicate how weights should be stored.
This struct implements the `Recorder` trait, which makes it capable of saving records for persistency.

We then build the learner with the model, the optimizer and the learning rate.
Notably, the third argument of the build function should actually be a learning rate _scheduler_. When provided with a float as in our example, it is automatically transformed into a _constant_ learning rate scheduler.
The learning rate is not part of the optimizer config as it is often done in other frameworks, but rather passed as a parameter when executing the optimizer step.
This avoids having to mutate the state of the optimizer and is therefore more functional.
It makes no difference when using the learner struct, but it will be an essential nuance to grasp if you implement your own training loop. 

Once the learner is created, we can simply call `fit` and provide the training and validation dataloaders.
For the sake of simplicity in this example, we employ the test set as the validation set; however, we do not recommend this practice for actual usage.

Finally, the trained model is returned by the `fit` method, and the only remaining task is saving the trained weights using the `CompactRecorder`.
This recorder employs the `MessagePack` format with `gzip` compression, `f16` for floats and `i32` for integers. Other recorders are available, offering support for various formats, such as `BinCode` and `JSON`, with or without compression. Any backend, regardless of precision, can load recorded data of any kind.
