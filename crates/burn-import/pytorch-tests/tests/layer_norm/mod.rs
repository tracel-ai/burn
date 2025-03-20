use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    norm1: LayerNorm<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let norm1 = LayerNormConfig::new(4).init(device);
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

    use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn layer_norm(record: NetRecord<Backend>, precision: usize) {
        let device = Default::default();

        let model = Net::<Backend>::init(&device).load_record(record);

        let input = Tensor::<Backend, 4>::from_data(
            [[
                [[0.757_631_6, 0.27931088], [0.40306926, 0.73468447]],
                [[0.02928156, 0.799_858_6], [0.39713734, 0.75437194]],
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = Tensor::<Backend, 4>::from_data(
            [[
                [[0.99991274, -0.999_912_5], [-0.999_818_3, 0.999_818_3]],
                [[-0.999_966_2, 0.99996626], [-0.99984336, 0.99984336]],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq(&expected.to_data(), precision);
    }

    #[test]
    fn layer_norm_full() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/layer_norm/layer_norm.pt".into(), &device)
            .expect("Should decode state successfully");
        layer_norm(record, 3);
    }

    #[test]
    fn layer_norm_half() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/layer_norm/layer_norm.pt".into(), &device)
            .expect("Should decode state successfully");
        layer_norm(record, 3);
    }
}
