use burn_core::nn::{
    gen_sequential, Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Relu,
};

use burn_core as burn;

gen_sequential! {
    Relu;
    DropoutConfig => Dropout,
    LeakyReluConfig => LeakyRelu;
    LinearConfig => Linear
}

type TestBackend = burn_ndarray::NdArray;

#[test]
fn sequential_should_construct() {
    let cfg = SequentialConfig {
        layers: vec![
            SequentialLayerConfig::Relu,
            SequentialLayerConfig::Dropout(DropoutConfig { prob: 0.3 }),
            SequentialLayerConfig::LeakyRelu(LeakyReluConfig {
                negative_slope: 0.01,
            }),
            SequentialLayerConfig::Linear(LinearConfig::new(10, 10)),
        ],
    };

    let device = Default::default();

    let module: Sequential<TestBackend> = cfg.init(&device);
    assert_eq!(module.layers.len(), 4);
}
