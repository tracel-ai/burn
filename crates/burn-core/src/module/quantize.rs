use burn_tensor::{
    Tensor,
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

impl ModuleMapper for Quantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        let tensor = tensor.quantize(&self.scheme, qparams);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(all(test, not(feature = "tch")))]
mod tests {
    use crate::TestDevice;
    use crate::module::{Module, Quantizer};
    use crate::test_utils::SimpleLinear;
    use burn_tensor::{
        Device, Tolerance,
        quantization::{Calibration, QuantLevel, QuantParam, QuantValue},
    };

    #[test]
    fn should_quantize_module() {
        let device = Device::new(TestDevice::default());
        let module = SimpleLinear::new(32, 32, &device);
        let scheme = device
            .default_quant_scheme()
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
