use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
struct Net<B: Backend> {
    norm1: LayerNorm<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        let norm1 = LayerNormConfig::new(4).init_with(record.norm1);
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

    use burn::record::{FullPrecisionSettings, Recorder, HalfPrecisionSettings};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn layer_norm(record: NetRecord<Backend>, precision: usize) {
        let device = Default::default();

        let model = Net::<Backend>::new_with(record);

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
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/layer_norm/layer_norm.pt".into())
            .expect("Failed to decode state");
        layer_norm(record, 3);
    }

    #[test]
    fn layer_norm_half() {
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/layer_norm/layer_norm.pt".into())
            .expect("Failed to decode state");
        layer_norm(record, 3);
    }
}
