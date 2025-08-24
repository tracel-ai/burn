use burn::{
    module::{Module, Param},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    buffer: Param<Tensor<B, 2>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        Self {
            buffer: record.buffer,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.buffer.val() + x
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::TestBackend;

    use burn::{
        record::{FullPrecisionSettings, Recorder},
        tensor::Tolerance,
    };
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    #[test]
    fn buffer() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/buffer/buffer.pt".into(), &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new_with(record);

        let input = Tensor::<TestBackend, 2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 2>::ones([3, 3], &device) * 2.0;

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }
}
