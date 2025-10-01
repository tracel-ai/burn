use burn::{
    module::Module,
    nn::{BatchNorm, BatchNormConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    norm1: BatchNorm<B>,
}

impl<B: Backend> Net<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            norm1: BatchNormConfig::new(4).init(device),
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.norm1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::{
        record::{FullPrecisionSettings, Recorder},
        tensor::Tolerance,
    };
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    #[test]
    fn batch_norm2d() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/batch_norm/batch_norm2d.pt".into(), &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new(&device).load_record(record);

        let input = Tensor::<TestBackend, 4>::ones([1, 5, 2, 2], &device) - 0.3;

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
                [[0.68515635, 0.68515635], [0.68515635, 0.68515635]],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
