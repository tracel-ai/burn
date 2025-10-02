use std::marker::PhantomData;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use guide::{
    data::{MnistBatch, MnistBatcher},
    model::{Model, ModelConfig},
};

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    // Create the batcher.
    let batcher = MnistBatcher::default();

    // Create the dataloaders.
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

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {} | Accuracy {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );
        }
    }
}

/// Create out own accuracy metric calculation.
fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze_dim(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}

#[allow(dead_code)]
struct Learner1<B, O>
where
    B: AutodiffBackend,
{
    model: Model<B>,
    optim: O,
}

#[allow(dead_code)]
struct Learner2<M, O> {
    model: M,
    optim: O,
}

#[allow(dead_code)]
struct Learner3<B, M, O> {
    model: M,
    optim: O,
    _b: PhantomData<B>,
}

#[allow(dead_code)]
impl<B, O> Learner1<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn step1(&mut self, _batch: MnistBatch<B>) {
        //
    }
}

#[allow(dead_code)]
impl<B, O> Learner2<Model<B>, O>
where
    B: AutodiffBackend,
    O: Optimizer<Model<B>, B>,
{
    pub fn step2(&mut self, _batch: MnistBatch<B>) {
        //
    }
}

#[allow(dead_code)]
impl<M, O> Learner2<M, O> {
    pub fn step3<B>(&mut self, _batch: MnistBatch<B>)
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
        O: Optimizer<M, B>,
    {
        //
    }
}
