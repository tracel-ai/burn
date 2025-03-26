use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{Int, Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    embed: Embedding<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(10, 3).init(device);
        Self { embed }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(x)
    }
}

#[cfg(test)]
mod tests {
    type Backend = burn_ndarray::NdArray<f32>;
    use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder};
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn embedding(record: NetRecord<Backend>, precision: usize) {
        let device = Default::default();

        let model = Net::<Backend>::init(&device).load_record(record);

        let input = Tensor::<Backend, 2, Int>::from_data([[1, 2, 4, 5], [4, 3, 2, 9]], &device);

        let output = model.forward(input);

        let expected = Tensor::<Backend, 3>::from_data(
            [
                [
                    [-1.609_484_9, -0.10016718, -0.609_188_9],
                    [-0.97977227, -1.609_096_3, -0.712_144_6],
                    [-0.22227049, 1.687_113_4, -0.32062083],
                    [-0.29934573, 1.879_345_7, -0.07213178],
                ],
                [
                    [-0.22227049, 1.687_113_4, -0.32062083],
                    [0.303_722, -0.777_314_3, -0.25145486],
                    [-0.97977227, -1.609_096_3, -0.712_144_6],
                    [-0.02878714, 2.357_111, -1.037_338_7],
                ],
            ],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq(&expected.to_data(), precision);
    }

    #[test]
    fn embedding_full_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/embedding/embedding.pt".into(), &device)
            .expect("Should decode state successfully");

        embedding(record, 3);
    }

    #[test]
    fn embedding_half_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/embedding/embedding.pt".into(), &device)
            .expect("Should decode state successfully");

        embedding(record, 3);
    }
}
