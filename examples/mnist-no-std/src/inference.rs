use crate::{proto::*, util};
use alloc::vec::Vec;
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, Recorder, RecorderError},
    tensor::ElementComparison,
};
use burn_no_std_tests::{mlp::MlpConfig, model::MnistConfig};

struct Model<B: Backend>(burn_no_std_tests::model::Model<B>);

impl<B: Backend> Model<B> {
    fn new(device: &B::Device, data: Vec<u8>) -> Result<Self, RecorderError> {
        let mlp_config = MlpConfig::new();
        let mnist_config = MnistConfig::new(mlp_config);
        let model = burn_no_std_tests::model::Model::new(&mnist_config, device);

        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(data, device)?;
        Ok(Self(model.load_record(record)))
    }
    fn infer(&self, device: &B::Device, img: &MnistImage) -> u8 {
        let tensor = util::image_to_tensor(device, img);
        let output = self.0.forward(tensor);
        let output = burn::tensor::activation::softmax(output, 1).into_data();
        assert_eq!(output.num_elements(), 10);
        output
            .iter::<f32>()
            .enumerate()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(index, _)| index)
            .unwrap() as u8
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
        assert!(model.is_none());

        model.replace(NoStdModel::new(&DEVICE, record.to_vec()).unwrap());
    }

    pub fn infer(image: &[u8]) -> u8 {
        let model = MODEL.lock();
        assert!(!model.is_none());

        model
            .as_ref()
            .unwrap()
            .infer(&DEVICE, image.try_into().unwrap())
    }
}
