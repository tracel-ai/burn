use burn_tensor::{
    Tensor,
    backend::Backend,
    quantization::{Calibration, QuantScheme, compute_q_params, compute_range},
};

use crate::module::{ModuleMapper, ParamId};

/// Describes how to quantize a module.
pub struct Quantizer {
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantScheme,
}

impl<B: Backend> ModuleMapper<B> for Quantizer {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        tensor.quantize(&self.scheme, qparams)
    }
}

#[cfg(all(test, not(feature = "test-tch")))]
mod tests {
    use crate::{
        TestBackend,
        module::{Module, Quantizer},
        nn::{
            Linear, LinearConfig,
            transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        },
    };
    use burn_tensor::{
        Device, Distribution, Tensor, Tolerance,
        ops::{FloatElem, QuantizedTensor},
        quantization::{
            Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantScheme, QuantValue,
        },
    };

    type B = TestBackend;

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
            .with_level(QuantLevel::Block(32))
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
            .with_level(QuantLevel::Block(16))
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
            .with_level(QuantLevel::Block(16))
            // .with_store(QuantStore::Native)
            .with_param(QuantParam::F32);

        should_quantize_module(
            transformer,
            scheme,
            |tr| tr.weight.val().dequantize(),
            Tolerance::permissive(),
        );
    }
}
