use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Net<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(2, 3).init(device);
        let fc2 = LinearConfig::new(3, 4).with_bias(false).init(device);
        let relu = Relu;

        Self { fc1, fc2, relu }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);

        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
struct NetWithBias<B: Backend> {
    fc1: Linear<B>,
}

impl<B: Backend> NetWithBias<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(2, 3).init(device);

        Self { fc1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.fc1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::tensor::{Tolerance, ops::FloatElem};
    use burn_store::{ModuleSnapshot, PytorchStore};
    type FT = FloatElem<TestBackend>;

    use super::*;

    fn linear_test(model: Net<TestBackend>, precision: f32) {
        let device = Default::default();

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
                [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
            ]],
            &device,
        );

        let output = model.forward(input);
        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [0.09778349, -0.13756673, 0.04962806, 0.08856435],
                    [0.03163241, -0.02848549, 0.01437942, 0.11905234],
                ],
                [
                    [0.07628226, -0.10757702, 0.03656857, 0.03824598],
                    [0.05443089, -0.06904714, 0.02744314, 0.09997337],
                ],
            ]],
            &device,
        );
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn linear_full_precision() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/linear/linear.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        linear_test(model, 1e-7);
    }

    #[test]
    fn linear_half_precision() {
        let device = Default::default();
        let mut model = Net::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/linear/linear.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        linear_test(model, 1e-4);
    }

    #[test]
    fn linear_with_bias() {
        let device = Default::default();

        let mut model = NetWithBias::<TestBackend>::init(&device);
        let mut store = PytorchStore::from_file("tests/linear/linear_with_bias.pt");
        model
            .load_from(&mut store)
            .expect("Should decode state successfully");

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
                [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [-0.00432095, -1.107_101_2, 0.870_691_4],
                    [0.024_595_5, -0.954_462_9, 0.48518157],
                ],
                [
                    [0.34315687, -0.757_384_2, 0.548_288],
                    [-0.06608963, -1.072_072_7, 0.645_800_5],
                ],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
