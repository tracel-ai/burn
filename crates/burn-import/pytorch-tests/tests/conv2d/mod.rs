use burn::{
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([2, 2], [2, 2]).init(device);
        let conv2 = Conv2dConfig::new([2, 2], [2, 2])
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

    use burn::{
        record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder},
        tensor::Tolerance,
    };
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn conv2d(record: NetRecord<TestBackend>, precision: f32) {
        let device = Default::default();

        let model = Net::<TestBackend>::init(&device).load_record(record);

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [
                        0.024_595_8,
                        0.25883394,
                        0.93905586,
                        0.416_715_5,
                        0.713_979_7,
                    ],
                    [0.267_644_3, 0.990_609, 0.28845078, 0.874_962_4, 0.505_920_8],
                    [0.23659128, 0.757_007_4, 0.23458993, 0.64705235, 0.355_621_4],
                    [0.445_182_8, 0.01930594, 0.26160914, 0.771_317, 0.37846136],
                    [
                        0.99802476,
                        0.900_794_2,
                        0.476_588_2,
                        0.16625845,
                        0.804_481_1,
                    ],
                ],
                [
                    [
                        0.65517855,
                        0.17679012,
                        0.824_772_3,
                        0.803_550_9,
                        0.943_447_5,
                    ],
                    [0.21972018, 0.417_697, 0.49031407, 0.57302874, 0.12054086],
                    [0.14518881, 0.772_002_3, 0.38275403, 0.744_236_7, 0.52850497],
                    [0.664_172_4, 0.60994434, 0.681_799_7, 0.74785537, 0.03694397],
                    [
                        0.751_675_7,
                        0.148_438_4,
                        0.12274551,
                        0.530_407_2,
                        0.414_796_4,
                    ],
                ],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [
                    [-0.02502128, 0.00250649, 0.04841233],
                    [0.04589614, -0.00296854, 0.01991477],
                    [0.02920526, 0.059_497_3, 0.04326791],
                ],
                [
                    [-0.04825336, 0.080_190_9, -0.02375088],
                    [0.02885434, 0.09638263, -0.07460806],
                    [0.02004079, 0.06244051, 0.035_887_1],
                ],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn conv2d_full_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/conv2d/conv2d.pt".into(), &device)
            .expect("Should decode state successfully");

        conv2d(record, 1e-7);
    }

    #[test]
    fn conv2d_half_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/conv2d/conv2d.pt".into(), &device)
            .expect("Should decode state successfully");

        conv2d(record, 1e-4);
    }
}
