use burn::{
    module::Module,
    nn::conv::{Conv1d, Conv1dConfig},
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
pub struct Net {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl Net {
    /// Create a new model from the given record.
    pub fn init(device: &Device) -> Self {
        let conv1 = Conv1dConfig::new(2, 2, 2).init(device);
        let conv2 = Conv1dConfig::new(2, 2, 2).with_bias(false).init(device);

        Self { conv1, conv2 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.conv1.forward(x);

        self.conv2.forward(x)
    }
}

#[cfg(test)]
mod tests {

    use burn::tensor::Tolerance;
    use burn_store::{ModuleSnapshot, PytorchStore};
    type FT = f32;

    use super::*;

    fn conv1d(model: Net, precision: f32) {
        let device = Default::default();

        let input = Tensor::<3>::from_data(
            [[
                [
                    0.93708336, 0.65559506, 0.31379688, 0.19801933, 0.41619217, 0.28432965,
                ],
                [
                    0.33977574,
                    0.523_940_8,
                    0.798_063_9,
                    0.77176833,
                    0.01122457,
                    0.80996025,
                ],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<3>::from_data(
            [[
                [0.02987457, 0.03134188, 0.04234261, -0.02437721],
                [-0.03788019, -0.02972012, -0.00806090, -0.01981254],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn conv1d_full_precision() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/conv1d/conv1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv1d(model, 1e-7);
    }

    #[test]
    fn conv1d_half_precision() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/conv1d/conv1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv1d(model, 1e-4);
    }
}
