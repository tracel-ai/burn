use burn::{
    module::Module,
    nn::{GroupNorm, GroupNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
struct Net<B: Backend> {
    norm1: GroupNorm<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let norm1 = GroupNormConfig::new(2, 6).init_with(record.norm1);
        Self { norm1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.norm1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    type Backend = burn_ndarray::NdArray<f32>;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    #[test]
    fn group_norm() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/group_norm/group_norm.pt".into())
            .expect("Failed to decode state");

        let model = Net::<Backend>::new_with(record);

        let input = Tensor::<Backend, 4>::from_data(
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

        let expected = Tensor::<Backend, 4>::from_data(
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

        output.to_data().assert_approx_eq(&expected.to_data(), 3);
    }
}
