use crate::{
    model::{mlp::MlpConfig, MnistConfig},
    proto::*,
    util,
};
use alloc::vec::Vec;
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, Recorder, RecorderError},
    tensor::cast::ToElement,
};

struct Model<B: Backend>(crate::model::Model<B>);

impl<B: Backend> Model<B> {
    fn new(device: &B::Device, data: Vec<u8>) -> Result<Self, RecorderError> {
        let mlp_config = MlpConfig::new();
        let mnist_config = MnistConfig::new(mlp_config);
        let model = crate::model::Model::new(&mnist_config, device);

        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(data, device)?;
        Ok(Self(model.load_record(record)))
    }
    fn infer(&self, device: &B::Device, img: &MnistImage) -> u8 {
        let tensor = util::image_to_tensor(device, img);
        let output = self.0.forward(tensor);
        let output = burn::tensor::activation::softmax(output, 1);
        output.argmax(1).into_scalar().to_u8()
    }
}

pub mod no_std_world {
    use super::Model;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use spin::Mutex;

    type NoStdModel = Model<NdArray>;

    const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
    static MODEL: Mutex<Option<NoStdModel>> = Mutex::new(Option::None);

    pub fn initialize(record: &[u8]) {
        let mut model = MODEL.lock();
        assert!(model.is_none(), "Model has been initialized");

        model.replace(NoStdModel::new(&DEVICE, record.to_vec()).unwrap());
    }

    pub fn infer(image: &[u8]) -> u8 {
        let model = MODEL.lock();
        assert!(!model.is_none());

        model
            .as_ref()
            .expect("Model has not been initialized")
            .infer(&DEVICE, image.try_into().unwrap())
    }
}
