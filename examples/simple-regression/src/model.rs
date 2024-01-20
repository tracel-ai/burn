use crate::dataset::DiabetesBatch;
use burn::config::Config;
use burn::nn::loss::Reduction::Mean;
use burn::nn::ReLU;
use burn::{
    module::Module,
    nn::{loss::MSELoss, Linear, LinearConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: ReLU,
}

#[derive(Config)]
pub struct RegressionModelConfig {
    pub num_features: usize,

    #[config(default = 64)]
    pub hidden_size: usize,
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let input_layer = LinearConfig::new(self.num_features, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);

        RegressionModel {
            input_layer,
            output_layer,
            activation: ReLU::new(),
        }
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: DiabetesBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MSELoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<DiabetesBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: DiabetesBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DiabetesBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: DiabetesBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
