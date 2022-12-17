use burn::optim::{decay::WeightDecayConfig, momentum::MomentumConfig};
use text_generation::{training::ExperimentConfig, DbPediaDataset};

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<burn::tensor::f16>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(512, 1024, 4, 4),
        burn::optim::SgdConfig::new()
            .with_learning_rate(5.0e-3)
            .with_weight_decay(Some(WeightDecayConfig::new(5e-4)))
            .with_momentum(Some(MomentumConfig::new().with_nesterov(true))),
    );

    text_generation::training::train::<Backend, DbPediaDataset>(
        burn_tch::TchDevice::Cuda(0),
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
