use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn_tch::{TchBackend, TchDevice};

use text_classification::{training::ExperimentConfig, DbPediaDataset};

#[cfg(not(feature = "f16"))]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

type Backend = burn_autodiff::ADBackendDecorator<TchBackend<ElemType>>;

fn main() {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            TchDevice::Mps
        } else {
            TchDevice::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-classification-db-pedia",
    );
}
