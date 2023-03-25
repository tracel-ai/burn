#[cfg(feature = "training")]
fn training() {
    use burn::optim::{decay::WeightDecayConfig, momentum::MomentumConfig};
    use text_classification::{training::ExperimentConfig, DbPediaDataset};

    type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<burn::tensor::f16>>;

    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        burn::optim::SgdConfig::new(5.0e-3)
            .with_momentum(Some(MomentumConfig::new().with_nesterov(true)))
            .with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
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

#[cfg(feature = "inference")]
fn inference() {
    use text_classification::DbPediaDataset;

    type Backend = burn_ndarray::NdArrayBackend<f32>;

    text_classification::inference::infer::<Backend, DbPediaDataset>(
        burn_ndarray::NdArrayDevice::Cpu,
        "/tmp/text-classification-nd-pedia",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            "Jays power up to take finale Contrary to popular belief, the power never really snapped back at SkyDome on Sunday. The lights came on after an hour delay, but it took some extra time for the batting orders to provide some extra wattage.".to_string(),
            "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one man to death and 14 others to prison terms for a series of attacks and terrorist plots in 2002, including the bombing of a French oil tanker.".to_string(),
            "IBM puts grids to work at U.S. Open IBM will put a collection of its On Demand-related products and technologies to this test next week at the U.S. Open tennis championships, implementing a grid-based infrastructure capable of running multiple workloads including two not associated with the tournament.".to_string(),
        ],
    );
}

fn main() {
    #[cfg(feature = "training")]
    training();
    #[cfg(feature = "inference")]
    inference();
}
