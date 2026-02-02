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
            norm1: BatchNormConfig::new(5).init(device), // Python model uses BatchNorm2d(5)
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

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    #[test]
    fn batch_norm2d() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::new(&device);
        let mut store = PytorchStore::from_file("tests/batch_norm/batch_norm2d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

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
