use burn::optim::{decay::WeightDecayConfig, momentum::MomentumConfig};
use text_classification::{training::ExperimentConfig, AgNewsDataset};

type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<f32>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(256, 512, 4, 4),
        burn::optim::SgdConfig::new()
            .with_learning_rate(5.0e-3)
            .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
            .with_momentum(Some(MomentumConfig::new().with_nesterov(true))),
    );

    text_classification::training::train::<Backend, AgNewsDataset>(
        burn_tch::TchDevice::Cuda(0),
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}
