use crate::{Tensor, TensorPrimitive, backend::Backend};
use burn_backend::tensor::quantization;

// We re-export those types.
pub use burn_backend::{QTensorPrimitive, quantization::*};

/// The tensor quantization parameters.
pub type QuantizationParameters<B> = QParams<Tensor<B, 1>>;

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange<B: Backend> {
    /// Minimum observed value(s).
    pub min: Tensor<B, 1>,
    /// Maximum observed value(s).
    pub max: Tensor<B, 1>,
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend, const D: usize>(
    scheme: &QuantScheme,
    tensor: &Tensor<B, D>,
    calibration: &Calibration,
) -> CalibrationRange<B> {
    let (min, max) = match &tensor.primitive {
        TensorPrimitive::Float(tensor) => {
            quantization::compute_range::<B>(scheme, tensor.clone(), calibration)
        }
        TensorPrimitive::QFloat(_) => unreachable!(),
    };

    CalibrationRange {
        min: Tensor::from_primitive(TensorPrimitive::Float(min)),
        max: Tensor::from_primitive(TensorPrimitive::Float(max)),
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    range: CalibrationRange<B>,
) -> QuantizationParameters<B> {
    match (range.min.primitive, range.max.primitive) {
        (TensorPrimitive::Float(min), TensorPrimitive::Float(max)) => {
            let qparams = quantization::compute_q_params::<B>(scheme, min, max);
            QuantizationParameters {
                scales: Tensor::from_primitive(TensorPrimitive::Float(qparams.scales)),
                zero_points: None,
            }
        }
        _ => unreachable!(),
    }
}
