use crate::dataset::DiabetesBatch;
use burn::{
    module::Module,
    nn::{loss::MSELoss, Linear, LinearConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use burn::nn::loss::Reduction::Mean;

#[derive(Module, Debug)]
pub struct LinearModel<B: Backend> {
    linear_layer: Linear<B>
}

impl<B: Backend> Default for LinearModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}


impl<B: Backend> LinearModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear_layer = LinearConfig::new(10, 1)
            .with_bias(true).init(device);

        Self {
            linear_layer
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        self.linear_layer.forward(x)
    }

    pub fn forward_regression(&self, item: DiabetesBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.inputs);
        // let loss = MSELoss::new()
        //     .init(&output.device())
        //     .forward(output.clone(), targets.clone());

        let loss = MSELoss::<B>::new()
            .forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}


impl<B: AutodiffBackend> TrainStep<DiabetesBatch<B>, RegressionOutput<B>> for LinearModel<B> {
    fn step(&self, item: DiabetesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DiabetesBatch<B>, RegressionOutput<B>> for LinearModel<B> {
    fn step(&self, item: DiabetesBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item)
    }
}
