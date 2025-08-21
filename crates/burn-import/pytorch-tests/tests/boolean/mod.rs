use burn::{
    module::{Module, Param},
    tensor::{Bool, Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    buffer: Param<Tensor<B, 1, Bool>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        Self {
            buffer: record.buffer,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1, Bool> {
        self.buffer.val()
    }
}

#[cfg(test)]
mod tests {

    use burn::{
        record::{FullPrecisionSettings, Recorder},
        tensor::TensorData,
    };
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    use crate::backend::TestBackend;

    #[test]
    #[ignore = "It appears loading boolean tensors are not supported yet"]
    // Error skipping: Msg("unsupported storage type BoolStorage")
    fn boolean() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/boolean/boolean.pt".into(), &device)
            .expect("Should decode state successfully");

        let model = Net::<TestBackend>::new_with(record);

        let input = Tensor::<TestBackend, 2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<TestBackend, 1, Bool>::from_bool(
            TensorData::from([true, false, true]),
            &device,
        );

        assert_eq!(output.to_data(), expected.to_data());
    }
}
