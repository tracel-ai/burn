use crate::dataset::{HousingBatch, NUM_FEATURES};
use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        loss::{MseLoss, Reduction::Mean},
    },
    prelude::*,
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

#[derive(Module, Debug)]
pub struct RegressionModel {
    input_layer: Linear,
    output_layer: Linear,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct RegressionModelConfig {
    #[config(default = 64)]
    pub hidden_size: usize,
}

impl RegressionModelConfig {
    pub fn init(&self, device: &Device) -> RegressionModel {
        let input_layer = LinearConfig::new(NUM_FEATURES, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);

        RegressionModel {
            input_layer,
            output_layer,
            activation: Relu::new(),
        }
    }
}

impl RegressionModel {
    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let x = self.input_layer.forward(input);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: HousingBatch) -> RegressionOutput {
        let targets: Tensor<2> = item.targets.unsqueeze_dim(1);
        let output: Tensor<2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl TrainStep for RegressionModel {
    type Input = HousingBatch;
    type Output = RegressionOutput;

    fn step(&self, item: HousingBatch) -> TrainOutput<RegressionOutput> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for RegressionModel {
    type Input = HousingBatch;
    type Output = RegressionOutput;

    fn step(&self, item: HousingBatch) -> RegressionOutput {
        self.forward_step(item)
    }
}
