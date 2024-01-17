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
use burn::nn::ReLU;

#[derive(Module, Debug)]
pub struct LinearModel<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: ReLU
}

impl<B: Backend> Default for LinearModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(10, &device)
    }
}


impl<B: Backend> LinearModel<B> {
    pub fn new(feature_len: usize, device: &B::Device) -> Self {
        let input_layer = LinearConfig::new(feature_len, 64)
            .with_bias(true).init(device);
        let output_layer = LinearConfig::new(64, 1)
            .with_bias(true).init(device);

        Self {
            input_layer,
            output_layer,
            activation: ReLU::new()
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        let prediction = self.output_layer.forward(x);
        prediction

    }

    pub fn forward_regression(&self, item: DiabetesBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MSELoss::new()
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
