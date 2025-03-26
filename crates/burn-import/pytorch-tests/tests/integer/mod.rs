use burn::{
    module::{Module, Param},
    tensor::{Int, Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    buffer: Param<Tensor<B, 1, Int>>,
}

impl<B: Backend> Net<B> {
    /// Create a new model from the given record.
    pub fn new_with(record: NetRecord<B>) -> Self {
        Self {
            buffer: record.buffer,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        self.buffer.val()
    }
}

#[cfg(test)]
mod tests {
    type Backend = burn_ndarray::NdArray<f32>;
    use burn::{
        record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder},
        tensor::TensorData,
    };
    use burn_import::pytorch::PyTorchFileRecorder;

    use super::*;

    fn integer(record: NetRecord<Backend>, _precision: usize) {
        let device = Default::default();

        let model = Net::<Backend>::new_with(record);

        let input = Tensor::<Backend, 2>::ones([3, 3], &device);

        let output = model.forward(input);

        let expected = Tensor::<Backend, 1, Int>::from_data(TensorData::from([1, 2, 3]), &device);

        assert_eq!(output.to_data(), expected.to_data());
    }

    #[test]
    fn integer_full_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load("tests/integer/integer.pt".into(), &device)
            .expect("Should decode state successfully");

        integer(record, 0);
    }

    #[test]
    fn integer_half_precision() {
        let device = Default::default();
        let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
            .load("tests/integer/integer.pt".into(), &device)
            .expect("Should decode state successfully");

        integer(record, 0);
    }
}
