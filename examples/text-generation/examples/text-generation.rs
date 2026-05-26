use burn::{
    optim::decay::WeightDecayConfig,
    tensor::{Device, DeviceConfig, DeviceIndex, Element},
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
        Device::libtorch_mps()
    } else {
        Device::libtorch_cuda(DeviceIndex::Default)
    };

    device
        .configure(DeviceConfig::default().float_dtype(Elem::dtype()))
        .unwrap();

    text_generation::training::train::<DbPediaDataset>(
        device,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
