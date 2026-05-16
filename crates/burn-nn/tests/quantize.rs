use burn_core as burn;

use burn::module::{Module, Quantizer};
use burn::tensor::{
    Distribution, Tensor, Tolerance,
    quantization::{Calibration, QuantLevel, QuantParam, QuantScheme, QuantValue},
};
use burn_nn::{
    LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};

fn should_quantize_module<M: Module, const D: usize, F: Fn(&M) -> Tensor<D>>(
    module: M,
    scheme: QuantScheme,
    func: F,
    tolerance: Tolerance<f32>,
) {
    let result = func(&module);

    let calibration = Calibration::MinMax;
    let mut quantizer = Quantizer {
        calibration,
        scheme,
    };
    let q_module = module.quantize_weights(&mut quantizer);
    let q_result = func(&q_module);

    result
        .into_data()
        .assert_approx_eq::<f32>(&q_result.into_data(), tolerance);
}

#[test]
fn should_quantize_transformer() {
    let device = Default::default();
    let transformer: TransformerEncoder =
        TransformerEncoderConfig::new(128, 256, 2, 2).init(&device);
    let signal = Tensor::random([2, 32, 128], Distribution::Default, &device);
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([32]))
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.forward(TransformerEncoderInput::new(signal.clone())),
        Tolerance::rel_abs(1e-2, 2e-2), // slightly higher abs tolerance (permissive: 1e-2)
    );
}

#[test]
fn should_quantize_linear_128_256() {
    let device = Default::default();
    let transformer = LinearConfig::new(128, 256).with_bias(false).init(&device);
    let signal = Tensor::<2>::random([1, 128], Distribution::Default, &device);
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Tensor)
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.forward(signal.clone()),
        Tolerance::permissive(),
    );
}

#[test]
fn should_quantize_linear() {
    let device = Default::default();
    let transformer = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let signal = Tensor::<2>::random([1, 32], Distribution::Default, &device);
    // Default scheme should select supported QuantStore default
    // TODO: set native if dtype is supported by the test backend
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Tensor)
        // .with_store(QuantStore::Native)
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.forward(signal.clone()),
        Tolerance::permissive(),
    );
}

#[test]
fn should_quantize_linear_weights() {
    let device = Default::default();
    let transformer = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Tensor)
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.weight.val().dequantize(),
        Tolerance::permissive(),
    );
}

#[test]
fn should_quantize_linear_blocks() {
    let device = Default::default();
    let transformer = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let signal = Tensor::<2>::random([1, 32], Distribution::Default, &device);
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([16]))
        // .with_store(QuantStore::Native)
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.forward(signal.clone()),
        Tolerance::permissive(),
    );
}

#[test]
fn should_quantize_linear_weights_blocks() {
    let device = Default::default();
    let transformer = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let scheme = device
        .default_quant_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([16]))
        // .with_store(QuantStore::Native)
        .with_param(QuantParam::F32);

    should_quantize_module(
        transformer,
        scheme,
        |tr| tr.weight.val().dequantize(),
        Tolerance::permissive(),
    );
}
