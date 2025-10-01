use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
    norm1: BatchNorm<B>,
    fc1: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Net<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([3, 4], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm1: BatchNormConfig::new(4).init(device),
            fc1: LinearConfig::new(4 * 8 * 8, 16).init(device),
            relu: Relu::new(),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = self.relu.forward(x);
        // Flatten all dimensions except the batch dimension
        let x = x.flatten(1, 3);
        self.fc1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::{
        record::{FullPrecisionSettings, Recorder},
        tensor::Tolerance,
    };
    use burn_import::safetensors::SafetensorsFileRecorder;

    use super::*;

    #[test]
    fn multi_layer_model() {
        let device = Default::default();
        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/multi_layer/multi_layer.safetensors".into(), &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new(&device).load_record(record);

        let input = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], &device);

        let output = model.forward(input);

        // Note: Expected values should be updated based on the actual output from the PyTorch model
        let expected = Tensor::<TestBackend, 2>::from_data(
            [[
                0.04971555,
                -0.16849735,
                0.05182848,
                -0.18032673,
                0.23138367,
                0.05041867,
                0.13005908,
                -0.32202929,
                -0.07915690,
                -0.03232457,
                -0.19790289,
                -0.17476529,
                -0.19627589,
                -0.21757686,
                -0.31376451,
                0.08377837,
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
