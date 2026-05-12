use burn::{
    optim::decay::WeightDecayConfig,
    tensor::{DType, Device, Element},
};
use text_generation::{DbPediaDataset, training::ExperimentConfig};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    let mut device: Device = if cfg!(target_os = "macos") {
        burn::backend::libtorch::LibTorchDevice::Mps.into()
    } else {
        burn::backend::libtorch::LibTorchDevice::Cuda(0).into()
    };

    device
        .set_default_dtypes(Elem::dtype(), DType::I64)
        .unwrap();

    text_generation::training::train::<DbPediaDataset>(
        device,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
