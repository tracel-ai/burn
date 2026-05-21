use crate::{
    Tensor,
    ops::{BridgeKind, BridgeTensor},
};
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
    let (kind, inner) = tensor.primitive.as_parts();
    let (min, max) = match kind {
        BridgeKind::Float => {
            quantization::compute_range::<Dispatch>(scheme, inner.clone(), calibration)
        }
        BridgeKind::QFloat => unreachable!(),
        _ => panic!("Should be Float primitive kind"),
    };

    CalibrationRange {
        min: Tensor::new(BridgeTensor::float(min)),
        max: Tensor::new(BridgeTensor::float(max)),
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params(scheme: &QuantScheme, range: CalibrationRange) -> QuantizationParameters {
    let (min_kind, min) = range.min.primitive.into_parts();
    let (max_kind, max) = range.max.primitive.into_parts();
    match (min_kind, max_kind) {
        (BridgeKind::Float, BridgeKind::Float) => {
            let qparams = quantization::compute_q_params::<Dispatch>(scheme, min, max);
            QuantizationParameters {
                scales: Tensor::new(BridgeTensor::float(qparams.scales)),
            }
        }
        _ => unreachable!(),
    }
}
