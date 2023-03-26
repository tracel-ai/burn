#[cfg(feature = "training")]
fn training() {
    use burn::optim::{decay::WeightDecayConfig, momentum::MomentumConfig};
    use text_classification::{training::ExperimentConfig, DbPediaDataset};

    #[cfg(not(feature = "f16"))]
    type ElemType = f32;
    #[cfg(feature = "f16")]
    type ElemType = burn::tensor::f16;

    type Backend = burn_autodiff::ADBackendDecorator<burn_tch::TchBackend<ElemType>>;

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
        "/tmp/text-classification-db-pedia",
        // Samples from the test dataset, but you are free to test with your own text.
        vec![
            " Magnus Eriksson is a Swedish former footballer who played as a forward.".to_string(),
            "Crossbeam Systems is headquartered in Boxborough Massachusetts and has offices in Europe Latin America and Asia Pacific. Crossbeam Systems was acquired by Blue Coat Systems in December 2012 and the Crossbeam brand has been fully absorbed into Blue Coat.".to_string(),
            " Zia is the sequel to the award-winning Island of the Blue Dolphins by Scott O'Dell. It was published in 1976 sixteen years after the publication of the first novel.".to_string(),
        ],
    );
}

fn main() {
    #[cfg(feature = "training")]
    training();
    #[cfg(feature = "inference")]
    inference();
}
