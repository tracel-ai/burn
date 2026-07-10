use burn_tensor::{
    Tensor,
    quantization::{Calibration, QuantScheme, compute_q_params, compute_range},
};

use crate::module::{ModuleMapper, Param, ParamGroup};

/// Describes how to quantize a module.
pub struct Quantizer {
    path: Vec<String>,
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The parameter group to quantize.
    pub group: ParamGroup,
}

impl Quantizer {
    /// Create a new [Quantizer].
    pub fn new(calibration: Calibration, scheme: QuantScheme) -> Self {
        Self {
            path: vec![],
            calibration,
            scheme,
            group: ParamGroup::all(),
        }
    }

    /// Set the parameter group to quantize.
    pub fn set_group(&mut self, group: ParamGroup) {
        self.group = group
    }
}

impl ModuleMapper for Quantizer {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, mut tensor, mapper) = param.consume();
        let path = self.path.join(".");
        if self.group.matches(&id, Some(&path)) {
            let range = compute_range(&self.scheme, &tensor, &self.calibration);
            let qparams = compute_q_params(&self.scheme, range);
            tensor = tensor.quantize(&self.scheme, qparams);
        }
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(all(test, not(feature = "tch")))]
mod tests {
    use crate::module::{Module, ParamGroup, Quantizer};
    use crate::tensor::DType;
    use crate::test_device;
    use crate::test_utils::SimpleLinear;
    use burn_tensor::{
        Device, Tensor, Tolerance,
        quantization::{Calibration, QuantLevel, QuantParam, QuantScheme, QuantValue},
    };

    /// Per-tensor Q8 symmetric scheme used across these tests.
    fn test_scheme(device: &Device) -> QuantScheme {
        device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32)
    }

    /// Whether a tensor currently holds quantized (`QFloat`) data.
    fn is_quantized<const D: usize>(tensor: &Tensor<D>) -> bool {
        matches!(tensor.dtype(), DType::QFloat(_))
    }

    #[test]
    fn should_quantize_module() {
        let device = test_device();
        let module = SimpleLinear::new(32, 32, &device);
        let scheme = test_scheme(&device);

        let result = module.weight.val();

        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer::new(calibration, scheme);
        let q_module = module.quantize_weights(&mut quantizer);
        let q_result = q_module.weight.val().dequantize();

        result
            .into_data()
            .assert_approx_eq::<f32>(&q_result.into_data(), Tolerance::permissive());
    }

    #[test]
    fn should_quantize_only_group() {
        let device = test_device();
        let module = SimpleLinear::new(32, 32, &device);
        let scheme = test_scheme(&device);

        let weight_ref = module.weight.val();

        let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);
        let q_module =
            module.quantize_weights_group(&mut quantizer, ParamGroup::from_path("weight"));

        // Only the weight (the group) is quantized; the bias is left untouched.
        assert!(is_quantized(&q_module.weight.val()));
        let bias = q_module.bias.clone().expect("bias should be present");
        assert!(!is_quantized(&bias.val()));

        // The quantized weight still approximates the original values.
        weight_ref.into_data().assert_approx_eq::<f32>(
            &q_module.weight.val().dequantize().into_data(),
            Tolerance::permissive(),
        );
    }

    #[test]
    fn should_quantize_all_when_no_group() {
        let device = test_device();
        let module = SimpleLinear::new(32, 32, &device);
        let scheme = test_scheme(&device);

        let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);
        let q_module = module.quantize_weights(&mut quantizer);

        // Without a group every float parameter is quantized.
        assert!(is_quantized(&q_module.weight.val()));
        assert!(is_quantized(&q_module.bias.clone().unwrap().val()));
    }
}
