use crate::{
    model::{mlp::MlpConfig, MnistConfig, Model},
    proto::*,
    util::{images_to_tensors, labels_to_tensors},
};
use alloc::vec::Vec;
use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{FullPrecisionSettings, Recorder, RecorderError},
    tensor::{backend::AutodiffBackend, cast::ToElement},
};

struct Trainer<B: AutodiffBackend> {
    model: Model<B>,
    device: B::Device,
    optim: OptimizerAdaptor<Adam, Model<B>, B>,
    lr: f64,
}

impl<B: AutodiffBackend> Trainer<B> {
    fn new(device: B::Device, seed: u64) -> Self {
        let config_optimizer = AdamConfig::new();
        let model_config = MnistConfig::new(MlpConfig::new()).with_seed(seed);

        B::seed(model_config.seed);

        Self {
            optim: config_optimizer.init(),
            model: Model::new(&model_config, &device),
            device,
            lr: 1e-4,
        }
    }

    fn with_learning_rate(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    // Originally inspired by burn/examples/custom-training-loop
    fn train(&mut self, images: &[MnistImage], labels: &[u8]) -> Output {
        let images = images_to_tensors(&self.device, images);
        let targets = labels_to_tensors(&self.device, labels);
        let model = self.model.clone();

        let output = model.forward(images);
        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());
        let accuracy = accuracy(output, targets);

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);
        // Update the model using the optimizer.
        self.model = self.optim.step(self.lr, model, grads);

        Output {
            loss: loss.into_scalar().to_f32(),
            accuracy,
        }
    }

    // Originally inspired by burn/examples/custom-training-loop
    fn valid(&self, images: &[MnistImage], labels: &[u8]) -> Output {
        // Get the model without autodiff.
        let model_valid = self.model.valid();

        let images = images_to_tensors(&self.device, images);
        let targets = labels_to_tensors(&self.device, labels);

        let output = model_valid.forward(images);
        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());
        let accuracy = accuracy(output, targets);

        Output {
            loss: loss.into_scalar().to_f32(),
            accuracy,
        }
    }

    fn export(&self) -> Result<Vec<u8>, RecorderError> {
        let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
        recorder.record(self.model.clone().into_record(), ())
    }
}

// Originally copy from burn/examples/custom-training-loop
/// Create out own accuracy metric calculation.
fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}

pub mod no_std_world {
    use super::Trainer;
    use crate::proto::*;
    use alloc::vec::Vec;
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    use spin::Mutex;

    type NoStdTrainer = Trainer<Autodiff<NdArray>>;

    const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
    static TRAINER: Mutex<Option<NoStdTrainer>> = Mutex::new(Option::None);

    pub fn initialize(seed: u64, lr: f64) {
        let mut trainer = TRAINER.lock();
        assert!(trainer.is_none(), "Trainer has been initialized");

        trainer.replace(NoStdTrainer::new(DEVICE, seed).with_learning_rate(lr));
    }

    pub fn train(images: &[u8], labels: &[u8]) -> Output {
        assert!(images.len() % MNIST_IMAGE_SIZE == 0);
        let images: &[MnistImage] = bytemuck::cast_slice(images);

        let mut trainer = TRAINER.lock();

        trainer
            .as_mut()
            .expect("Trainer has not been initialized")
            .train(images, labels)
    }

    pub fn valid(images: &[u8], labels: &[u8]) -> Output {
        assert!(images.len() % MNIST_IMAGE_SIZE == 0);
        let images: &[MnistImage] = bytemuck::cast_slice(images);

        let trainer = TRAINER.lock();

        trainer
            .as_ref()
            .expect("Trainer has not been initialized")
            .valid(images, labels)
    }

    pub fn export() -> Vec<u8> {
        let trainer = TRAINER.lock();

        trainer
            .as_ref()
            .expect("Trainer has not been initialized")
            .export()
            .unwrap()
    }
}
