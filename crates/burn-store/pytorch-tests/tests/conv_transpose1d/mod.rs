use burn::{
    module::Module,
    nn::conv::{ConvTranspose1d, ConvTranspose1dConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: ConvTranspose1d<B>,
    conv2: ConvTranspose1d<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = ConvTranspose1dConfig::new([2, 2], 2).init(device);
        let conv2 = ConvTranspose1dConfig::new([2, 2], 2)
            .with_bias(false)
            .init(device);

        Self { conv1, conv2 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv1.forward(x);

        self.conv2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};

    use super::*;

    fn conv_transpose1d(model: Net<TestBackend>, precision: f32) {
        let device = Default::default();

        let input = Tensor::<TestBackend, 3>::from_data(
            [[[0.93708336, 0.65559506], [0.31379688, 0.19801933]]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.02935525, 0.01119324, -0.01356167, -0.00682688],
                [0.01644749, -0.01429807, 0.00083987, 0.00279229],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn conv_transpose1d_full() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose1d/conv_transpose1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose1d(model, 1e-8);
    }

    #[test]
    fn conv_transpose1d_half() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose1d/conv_transpose1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose1d(model, 1e-4);
    }
}
