// Generated from ONNX "./onnx-tests/tests/reshape/reshape.onnx" by burn-import
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    phantom: core::marker::PhantomData<B>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("./out/reshape", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new_with(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new_with(record: ModelRecord<B>) -> Self {
        Self {
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(dead_code, unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        Self {
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 1>) -> Tensor<B, 2> {
        let reshape1_out1 = input1.reshape([2, 2]);
        let reshape2_out1 = reshape1_out1.reshape([1, -1]);
        reshape2_out1
    }
}
