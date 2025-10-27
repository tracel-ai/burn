# Training

We are now ready to write the necessary code to train our model on the MNIST dataset. We shall
define the code for this training section in the file: `src/training.rs`.

Instead of a simple tensor, the model should output an item that can be understood by the learner, a
struct whose responsibility is to apply an optimizer to the model. The output struct is used for all
metrics calculated during the training. Therefore it should include all the necessary information to
calculate any metric that you want for a task.

Burn provides two basic output types: `ClassificationOutput` and `RegressionOutput`. They implement
the necessary trait to be used with metrics. It is possible to create your own item, but it is
beyond the scope of this guide.

Since the MNIST task is a classification problem, we will use the `ClassificationOutput` type.

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
```

As evident from the preceding code block, we employ the cross-entropy loss module for loss
calculation, without the inclusion of any padding token. We then return the classification output
containing the loss, the output tensor with all logits and the targets.

Please take note that tensor operations receive owned tensors as input. For reusing a tensor
multiple times, you need to use the `clone()` function. There's no need to worry; this process won't
involve actual copying of the tensor data. Instead, it will simply indicate that the tensor is
employed in multiple instances, implying that certain operations won't be performed in place. In
summary, our API has been designed with owned tensors to optimize performance.

Moving forward, we will proceed with the implementation of both the training and validation steps
for our model.

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
# impl<B: Backend> Model<B> {
#     pub fn forward_classification(
#         &self,
#         images: Tensor<B, 3>,
#         targets: Tensor<B, 1, Int>,
#     ) -> ClassificationOutput<B> {
#         let output = self.forward(images);
#         let loss = CrossEntropyLossConfig::new()
#             .init(&output.device())
#             .forward(output.clone(), targets.clone());
# 
#         ClassificationOutput::new(loss, output, targets)
#     }
# }
# 
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
```

Here we define the input and output types as generic arguments in the `TrainStep` and `ValidStep`.
We will call them `MnistBatch` and `ClassificationOutput`. In the training step, the computation of
gradients is straightforward, necessitating a simple invocation of `backward()` on the loss. Note
that contrary to PyTorch, gradients are not stored alongside each tensor parameter, but are rather
returned by the backward pass, as such: `let gradients = loss.backward();`. The gradient of a
parameter can be obtained with the grad function: `let grad = tensor.grad(&gradients);`. Although it
is not necessary when using the learner struct and the optimizers, it can prove to be quite useful
when debugging or writing custom training loops. One of the differences between the training and the
validation steps is that the former requires the backend to implement `AutodiffBackend` and not just
`Backend`. Otherwise, the `backward` function is not available, as the backend does not support
autodiff. We will see later how to create a backend with autodiff support.

<details>
<summary><strong>🦀 Generic Type Constraints in Method Definitions</strong></summary>

Although generic data types, trait and trait bounds were already introduced in previous sections of
this guide, the previous code snippet might be a lot to take in at first.

In the example above, we implement the `TrainStep` and `ValidStep` trait for our `Model` struct,
which is generic over the `Backend` trait as has been covered before. These traits are provided by
`burn::train` and define a common `step` method that should be implemented for all structs. Since
the trait is generic over the input and output types, the trait implementation must specify the
concrete types used. This is where the additional type constraints appear
`<MnistBatch<B>, ClassificationOutput<B>>`. As we saw previously, the concrete input type for the
batch is `MnistBatch`, and the output of the forward pass is `ClassificationOutput`. The `step`
method signature matches the concrete input and output types.

For more details specific to constraints on generic types when defining methods, take a look at
[this section](https://doc.rust-lang.org/book/ch10-01-syntax.html#in-method-definitions) of the Rust
Book.

</details><br>

Let us move on to establishing the practical training configuration.

```rust , ignore
# use crate::{
#     data::{MnistBatch, MnistBatcher},
#     model::{Model, ModelConfig},
# };
# use burn::{
#     data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
#     nn::loss::CrossEntropyLossConfig,
#     optim::AdamConfig,
#     prelude::*,
#     record::CompactRecorder,
#     tensor::backend::AutodiffBackend,
#     train::{
#         metric::{AccuracyMetric, LossMetric},
#         ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
#     },
# };
# 
# impl<B: Backend> Model<B> {
#     pub fn forward_classification(
#         &self,
#         images: Tensor<B, 3>,
#         targets: Tensor<B, 1, Int>,
#     ) -> ClassificationOutput<B> {
#         let output = self.forward(images);
#         let loss = CrossEntropyLossConfig::new()
#             .init(&output.device())
#             .forward(output.clone(), targets.clone());
# 
#         ClassificationOutput::new(loss, output, targets)
#     }
# }
# 
# impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
#     fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
#         let item = self.forward_classification(batch.images, batch.targets);
# 
#         TrainOutput::new(self, item.loss.backward(), item)
#     }
# }
# 
# impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
#     fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
#         self.forward_classification(batch.images, batch.targets)
#     }
# }
# 
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    if let Some(summary) = result.summary {
        println!("{summary}");
    }
}
```

It is a good practice to use the `Config` derive to create the experiment configuration. In the
`train` function, the first thing we are doing is making sure the `artifact_dir` exists, using the
standard rust library for file manipulation. All checkpoints, logging and metrics will be stored
under this directory. We initialize the dataloaders using the previously created batcher. Since no
automatic differentiation is needed during the validation phase, the `learner.fit(...)` method
defines the necessary backend bounds on the data loader for `B::InnerBackend` (see
[Backend](./backend.md)). The autodiff capabilities are available through a type system, making it
nearly impossible to forget to deactivate gradient calculation.

Next, we create our learner with the accuracy and loss metric on both training and validation steps
along with the device and the epoch. We also configure the checkpointer using the `CompactRecorder`
to indicate how weights should be stored. This struct implements the `Recorder` trait, which makes
it capable of saving records for persistency.

We then build the learner with the model, the optimizer and the learning rate. Notably, the third
argument of the build function should actually be a learning rate _scheduler_. When provided with a
float as in our example, it is automatically transformed into a _constant_ learning rate scheduler.
The learning rate is not part of the optimizer config as it is often done in other frameworks, but
rather passed as a parameter when executing the optimizer step. This avoids having to mutate the
state of the optimizer and is therefore more functional. It makes no difference when using the
learner struct, but it will be an essential nuance to grasp if you implement your own training loop.

Once the learner is created, we can simply call `fit` and provide the training and validation
dataloaders. For the sake of simplicity in this example, we employ the test set as the validation
set; however, we do not recommend this practice for actual usage.

Finally, the trained model is returned by the `fit` method. The trained weights are then saved using
the `CompactRecorder`. This recorder employs the `MessagePack` format with half precision, `f16` for
floats and `i16` for integers. Other recorders are available, offering support for various formats,
such as `BinCode` and `JSON`, with or without compression. Any backend, regardless of precision, can
load recorded data of any kind.
