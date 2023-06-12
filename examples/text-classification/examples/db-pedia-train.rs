use burn::optim::decay::WeightDecayConfig;
use text_classification::{training::ExperimentConfig, DbPediaDataset};

#[cfg(not(feature = "f16"))]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<ElemType>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            burn_tch::TchDevice::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-classification-db-pedia",
    );
}
