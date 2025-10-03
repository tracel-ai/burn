use burn_core as burn;

use burn::module::{Module, Quantizer};
use burn::tensor::{
    Device, Distribution, Tensor, Tolerance,
    ops::{FloatElem, QuantizedTensor},
    quantization::{
        Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantScheme, QuantValue,
    },
};
use burn_nn::{
    Linear, LinearConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};

#[cfg(all(
    test,
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type B = burn_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type B = burn_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type B = burn_cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type B = burn_rocm::Rocm;

fn should_quantize_module<M: Module<B>, const D: usize, F: Fn(&M) -> Tensor<B, D>>(
    module: M,
    scheme: QuantScheme,
    func: F,
    tolerance: Tolerance<FloatElem<B>>,
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
    let device: Device<B> = Default::default();
    let transformer: TransformerEncoder<B> =
        TransformerEncoderConfig::new(128, 256, 2, 2).init(&device);
    let signal = Tensor::random([2, 32, 128], Distribution::Default, &device);
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
    let device: Device<B> = Default::default();
    let transformer: Linear<B> = LinearConfig::new(128, 256).with_bias(false).init(&device);
    let signal = Tensor::<B, 2>::random([1, 128], Distribution::Default, &device);
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
    let device: Device<B> = Default::default();
    let transformer: Linear<B> = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let signal = Tensor::<B, 2>::random([1, 32], Distribution::Default, &device);
    // Default scheme should select supported QuantStore default
    // TODO: set native if dtype is supported by the test backend
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
    let device: Device<B> = Default::default();
    let transformer: Linear<B> = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
    let device: Device<B> = Default::default();
    let transformer: Linear<B> = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let signal = Tensor::<B, 2>::random([1, 32], Distribution::Default, &device);
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
    let device: Device<B> = Default::default();
    let transformer: Linear<B> = LinearConfig::new(32, 32).with_bias(false).init(&device);
    let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
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
