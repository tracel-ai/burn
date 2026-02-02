use burn::{
    module::Module,
    nn::conv::{ConvTranspose2d, ConvTranspose2dConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: ConvTranspose2d<B>,
    conv2: ConvTranspose2d<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = ConvTranspose2dConfig::new([2, 2], [2, 2]).init(device);
        let conv2 = ConvTranspose2dConfig::new([2, 2], [2, 2])
            .with_bias(false)
            .init(device);

        Self { conv1, conv2 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
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

    fn conv_transpose2d(model: Net<TestBackend>, precision: f32) {
        let device = Default::default();

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
                [[0.713_979_7, 0.267_644_3], [0.990_609, 0.28845078]],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.04547675, 0.01879685, -0.01636661, 0.00310803],
                    [0.02090115, 0.01192738, -0.048_240_2, 0.02252235],
                    [0.03249975, -0.00460748, 0.05003899, 0.04029131],
                    [0.02185687, -0.10226749, -0.06508022, -0.01267705],
                ],
                [
                    [0.00277598, -0.00513832, -0.059_048_3, 0.00567626],
                    [-0.03149522, -0.195_757_4, 0.03474613, 0.01997269],
                    [-0.10096474, 0.00679589, 0.041_919_7, -0.02464108],
                    [-0.03174751, 0.02963913, -0.02703723, -0.01860938],
                ],
            ]],
            &device,
        );
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn conv_transpose2d_full() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose2d/conv_transpose2d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose2d(model, 1e-7);
    }

    #[test]
    fn conv_transpose2d_half() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose2d/conv_transpose2d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose2d(model, 1e-4);
    }
}
