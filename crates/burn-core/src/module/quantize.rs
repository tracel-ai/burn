use burn_tensor::{
    Tensor,
    backend::Backend,
    quantization::{Calibration, QuantScheme, compute_q_params, compute_range},
};

use crate::module::{ModuleMapper, Param};

/// Describes how to quantize a module.
pub struct Quantizer {
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantScheme,
}

impl<B: Backend> ModuleMapper<B> for Quantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        let tensor = tensor.quantize(&self.scheme, qparams);
        Param::into_initialized(id, tensor, mapper)
    }
}

#[cfg(all(test, not(feature = "test-tch")))]
mod tests {
    use crate::test_utils::SimpleLinear;
    use crate::{
        TestBackend,
        module::{Module, Quantizer},
    };
    use burn_tensor::{
        Device, Tolerance,
        ops::QuantizedTensor,
        quantization::{Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantValue},
    };

    type B = TestBackend;

    #[test]
    fn should_quantize_module() {
        let device: Device<B> = Default::default();
        let module = SimpleLinear::<B>::new(32, 32, &device);
        let scheme = <QuantizedTensor<B> as QTensorPrimitive>::default_scheme()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32);

        let result = module.weight.val();

        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let q_module = module.quantize_weights(&mut quantizer);
        let q_result = q_module.weight.val().dequantize();

        result
            .into_data()
            .assert_approx_eq::<f32>(&q_result.into_data(), Tolerance::permissive());
    }
}
