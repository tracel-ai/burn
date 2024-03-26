use crate::dataset::StockBatch;
use burn::{
    nn::{
        loss::{MseLoss, Reduction::Mean},
        Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct ForecastingModel<B: Backend> {
    lstm: Lstm<B>,
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
    dropout: Dropout,
    num_features: usize,
    hidden_size: usize,
}

#[derive(Config)]
pub struct ForecastingModelConfig {
    pub num_features: usize,

    #[config(default = 64)]
    pub hidden_size: usize,
}

impl ForecastingModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ForecastingModel<B> {
        let input_layer = LinearConfig::new(self.num_features, self.hidden_size)
            .with_bias(true)
            .init(device);

        let activation = Relu::new();
        let lstm = LstmConfig::new(self.hidden_size, self.hidden_size, true).init(device);
        let dropout = DropoutConfig::new(0.2).init();
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);

        ForecastingModel {
            input_layer,
            lstm,
            output_layer,
            dropout,
            activation,
            num_features: self.num_features,
            hidden_size: self.hidden_size,
        }
    }
}

impl<B: Backend> ForecastingModel<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = input.detach();
        let batch_size = x.shape().dims[0];

        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);

        // Initialize cell state
        let s = Tensor::zeros([batch_size, self.hidden_size], &<B>::Device::default());
        // Initialize hidden state
        let h = Tensor::zeros([batch_size, self.hidden_size], &<B>::Device::default());

        // Run inputs through LSTM
        let x = self.lstm.forward(x, Some((s.detach(), h.detach())));

        let hidden = x.1;
        let sequence_size = hidden.shape().dims[1];
        let hidden_size = hidden.shape().dims[2];

        // Get last time steps hidden states
        let x = self.dropout.forward(
            hidden
                .slice([
                    0..batch_size,
                    sequence_size - 1..sequence_size,
                    0..hidden_size,
                ])
                .reshape([batch_size, hidden_size]),
        );

        self.output_layer.forward(x).reshape([batch_size, 1])
    }

    pub fn forward_step(&self, item: StockBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item
            .targets
            .clone()
            .reshape([item.targets.dims()[0] as i32, 1]);

        let output: Tensor<B, 2> = self.forward(item.inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<StockBatch<B>, RegressionOutput<B>> for ForecastingModel<B> {
    fn step(&self, item: StockBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<StockBatch<B>, RegressionOutput<B>> for ForecastingModel<B> {
    fn step(&self, item: StockBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
