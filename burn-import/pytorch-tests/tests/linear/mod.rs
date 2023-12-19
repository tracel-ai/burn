use burn::{
    module::Module,
    nn::{Linear, LinearConfig, ReLU},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
struct Net<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: ReLU,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let fc1 = LinearConfig::new(2, 3).init_with(record.fc1);
        let fc2 = LinearConfig::new(3, 4).init_with(record.fc2);
        let relu = ReLU::default();

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
    /// Create a new model from the given record.
    pub fn new_with(record: NetWithBiasRecord<B>) -> Self {
        let fc1 = LinearConfig::new(2, 3).init_with(record.fc1);

        Self { fc1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.fc1.forward(x)
    }
}

#[cfg(test)]
mod tests {
    type Backend = burn_ndarray::NdArray<f32>;

    use std::{env, path::Path};

    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    use super::*;

    #[test]
    fn linear() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/labeled/linear");

        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");

        let model = Net::<Backend>::new_with(record);

        let input = Tensor::<Backend, 4>::from_data([[
            [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
            [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
        ]]);

        let output = model.forward(input);

        let expected = Tensor::<Backend, 4>::from_data([[
            [
                [0.09778349, -0.13756673, 0.04962806, 0.08856435],
                [0.03163241, -0.02848549, 0.01437942, 0.11905234],
            ],
            [
                [0.07628226, -0.10757702, 0.03656857, 0.03824598],
                [0.05443089, -0.06904714, 0.02744314, 0.09997337],
            ],
        ]]);

        output.to_data().assert_approx_eq(&expected.to_data(), 6);
    }

    #[test]
    fn linear_with_bias() {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let file_path = Path::new(&out_dir).join("model/guessed/linear_with_bias");

        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(file_path)
            .expect("Failed to decode state");

        let model = NetWithBias::<Backend>::new_with(record);

        let input = Tensor::<Backend, 4>::from_data([[
            [[0.63968194, 0.97427773], [0.830_029_9, 0.04443115]],
            [[0.024_595_8, 0.25883394], [0.93905586, 0.416_715_5]],
        ]]);

        let output = model.forward(input);

        let expected = Tensor::<Backend, 4>::from_data([[
            [
                [-0.00432095, -1.107_101_2, 0.870_691_4],
                [0.024_595_5, -0.954_462_9, 0.48518157],
            ],
            [
                [0.34315687, -0.757_384_2, 0.548_288],
                [-0.06608963, -1.072_072_7, 0.645_800_5],
            ],
        ]]);

        output.to_data().assert_approx_eq(&expected.to_data(), 6);
    }
}
