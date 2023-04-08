use burn::optim::decay::WeightDecayConfig;
use text_generation::{training::ExperimentConfig, DbPediaDataset};

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<burn::tensor::f16>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(512, 2048, 16, 8)
            .with_norm_first(true),
        burn::optim::AdamConfig::new()
            .with_epsilon(1e-4)
            .with_weight_decay(Some(WeightDecayConfig::new(1.0e-5))),
    );

    text_generation::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            burn_tch::TchDevice::Mps
        } else {
            burn_tch::TchDevice::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
