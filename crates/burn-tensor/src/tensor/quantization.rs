use crate::{Tensor, TensorPrimitive};
use burn_backend::quantization;

// User-facing quantization data types come from burn-std.
use burn_dispatch::Dispatch;
pub use burn_std::quantization::*;

/// The tensor quantization parameters.
pub type QuantizationParameters = QParams<Tensor<1>>;

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange {
    /// Minimum observed value(s).
    pub min: Tensor<1>,
    /// Maximum observed value(s).
    pub max: Tensor<1>,
}

/// Compute the quantization range mapping.
pub fn compute_range<const D: usize>(
    scheme: &QuantScheme,
    tensor: &Tensor<D>,
    calibration: &Calibration,
) -> CalibrationRange {
    let (min, max) = match &tensor.primitive {
        TensorPrimitive::Float(tensor) => {
            quantization::compute_range::<Dispatch>(scheme, tensor.clone(), calibration)
        }
        TensorPrimitive::QFloat(_) => unreachable!(),
    };

    CalibrationRange {
        min: Tensor::from_primitive(TensorPrimitive::Float(min)),
        max: Tensor::from_primitive(TensorPrimitive::Float(max)),
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params(scheme: &QuantScheme, range: CalibrationRange) -> QuantizationParameters {
    match (range.min.primitive, range.max.primitive) {
        (TensorPrimitive::Float(min), TensorPrimitive::Float(max)) => {
            let qparams = quantization::compute_q_params::<Dispatch>(scheme, min, max);
            QuantizationParameters {
                scales: Tensor::from_primitive(TensorPrimitive::Float(qparams.scales)),
            }
        }
        _ => unreachable!(),
    }
}
