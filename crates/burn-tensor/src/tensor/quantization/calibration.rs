use crate::{Tensor, backend::Backend};

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange<B: Backend> {
    /// Minimum observed value(s).
    pub min: Tensor<B, 1>,
    /// Maximum observed value(s).
    pub max: Tensor<B, 1>,
}

/// Calibration method used to compute the quantization range mapping.
pub enum Calibration {
    /// Computes quantization range mapping based on the min and max values.
    MinMax,
}
