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

    use std::{env, path::Path};

    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    use super::*;

    #[test]
    #[ignore = "Failing possibly due to bug in Burn's group norm."]
    fn group_norm() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/group_norm");

        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");

        let model = Net::<Backend>::new_with(record);

        let input = Tensor::<Backend, 4>::from_data([[
            [[0.757_631_6, 0.27931088], [0.40306926, 0.73468447]],
            [[0.02928156, 0.799_858_6], [0.39713734, 0.75437194]],
            [[0.569_508_5, 0.43877792], [0.63868046, 0.524_665_9]],
            [[0.682_614_1, 0.305_149_5], [0.46354562, 0.45498633]],
            [[0.572_472, 0.498_002_6], [0.93708336, 0.65559506]],
            [[0.31379688, 0.19801933], [0.41619217, 0.28432965]],
        ]]);

        let output = model.forward(input);

        let expected = Tensor::<Backend, 4>::from_data([[
            [[1.042_578_5, -1.122_016_7], [-0.56195974, 0.938_733_6]],
            [[-2.253_500_7, 1.233_672_9], [-0.588_804_1, 1.027_827_3]],
            [[0.19124532, -0.40036356], [0.504_276_5, -0.01168585]],
            [[1.013_829_2, -0.891_984_6], [-0.09224463, -0.13546038]],
            [[0.45772314, 0.08172822], [2.298_641_4, 0.877_410_4]],
            [[-0.84832406, -1.432_883_4], [-0.331_331_5, -0.997_103_7]],
        ]]);

        output.to_data().assert_approx_eq(&expected.to_data(), 3);
    }
    // Current Error:
    //     ---- group_norm::tests::group_norm stdout ----
    // thread 'group_norm::tests::group_norm' panicked at burn-import/pytorch-tests/tests/group_norm/mod.rs:67:26:
    // Tensors are not approx eq:
    //   => Position 0: 0.275427907705307 != 1.0425784587860107 | difference 0.7671505510807037 > tolerance 0.0010000000000000002
    //   => Position 1: -0.29641395807266235 != -1.1220166683197021 | difference 0.8256027102470398 > tolerance 0.0010000000000000002
    //   => Position 2: -0.14845837652683258 != -0.5619597434997559 | difference 0.4135013669729233 > tolerance 0.0010000000000000002
    //   => Position 3: 0.24799413979053497 != 0.9387335777282715 | difference 0.6907394379377365 > tolerance 0.0010000000000000002
    //   => Position 4: -0.5953289270401001 != -2.2535006999969482 | difference 1.6581717729568481 > tolerance 0.0010000000000000002
    // 19 more errors...
    // note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
}
