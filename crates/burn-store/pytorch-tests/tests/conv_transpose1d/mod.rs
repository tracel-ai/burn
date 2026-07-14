use burn::{
    module::Module,
    nn::conv::{ConvTranspose1d, ConvTranspose1dConfig},
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
pub struct Net {
    conv1: ConvTranspose1d,
    conv2: ConvTranspose1d,
}

impl Net {
    /// Create a new model from the given record.
    pub fn init(device: &Device) -> Self {
        let conv1 = ConvTranspose1dConfig::new([2, 2], 2).init(device);
        let conv2 = ConvTranspose1dConfig::new([2, 2], 2)
            .with_bias(false)
            .init(device);

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

    use super::*;

    fn conv_transpose1d(model: Net, precision: f32) {
        let device = Default::default();

        let input = Tensor::<3>::from_data(
            [[[0.93708336, 0.65559506], [0.31379688, 0.19801933]]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<3>::from_data(
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
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose1d/conv_transpose1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose1d(model, 1e-8);
    }

    #[test]
    fn conv_transpose1d_half() {
        let device = Default::default();
        let mut model = Net::init(&device);
        let mut store = PytorchStore::from_file("tests/conv_transpose1d/conv_transpose1d.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        conv_transpose1d(model, 1e-4);
    }
}
