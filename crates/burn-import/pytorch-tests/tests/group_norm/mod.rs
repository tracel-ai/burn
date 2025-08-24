use burn::{
    module::Module,
    nn::{GroupNorm, GroupNormConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    norm1: GroupNorm<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let norm1 = GroupNormConfig::new(2, 6).init(device);
        Self { norm1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.norm1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;
    use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder};
    use burn::tensor::Tolerance;
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn group_norm(record: NetRecord<TestBackend>, precision: f32) {
        let device = Default::default();

        let model = Net::<TestBackend>::init(&device).load_record(record);

        let input = Tensor::<TestBackend, 4>::from_data(
            [[
                [[0.757_631_6, 0.27931088], [0.40306926, 0.73468447]],
                [[0.02928156, 0.799_858_6], [0.39713734, 0.75437194]],
                [[0.569_508_5, 0.43877792], [0.63868046, 0.524_665_9]],
                [[0.682_614_1, 0.305_149_5], [0.46354562, 0.45498633]],
                [[0.572_472, 0.498_002_6], [0.93708336, 0.65559506]],
                [[0.31379688, 0.19801933], [0.41619217, 0.28432965]],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 4>::from_data(
            [[
                [[1.042_578_5, -1.122_016_7], [-0.56195974, 0.938_733_6]],
                [[-2.253_500_7, 1.233_672_9], [-0.588_804_1, 1.027_827_3]],
                [[0.19124532, -0.40036356], [0.504_276_5, -0.01168585]],
                [[1.013_829_2, -0.891_984_6], [-0.09224463, -0.13546038]],
                [[0.45772314, 0.08172822], [2.298_641_4, 0.877_410_4]],
                [[-0.84832406, -1.432_883_4], [-0.331_331_5, -0.997_103_7]],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(precision));
    }

    #[test]
    fn group_norm_full() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/group_norm/group_norm.pt".into(), &device)
            .expect("Should decode state successfully");

        group_norm(record, 1e-3);
    }

    #[test]
    fn group_norm_half() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/group_norm/group_norm.pt".into(), &device)
            .expect("Should decode state successfully");

        group_norm(record, 1e-3);
    }
}
