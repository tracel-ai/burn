use crate::{backend::Backend, Tensor};

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange<B: Backend> {
    /// Minimum observed value.
    pub min: Tensor<B, 1>,
    /// Maximum observed value.
    pub max: Tensor<B, 1>,
}

/// Calibration method used to compute the quantization range mapping.
pub trait Calibration {
    /// Compute the input tensor range.
    fn compute_range<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> CalibrationRange<B>;
}

/// Computes the per-tensor quantization range mapping based on the min and max values.
pub struct MinMaxCalibration {}

impl Calibration for MinMaxCalibration {
    fn compute_range<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> CalibrationRange<B> {
        let min = tensor.clone().min();
        let max = tensor.clone().max();

        CalibrationRange { min, max }
    }
}
