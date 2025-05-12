use burn_tensor::{DType, Shape};

use crate::CubeRuntime;

use super::CubeTensor;

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QParams {
    /// Start of the scales tensor in the buffer
    pub scales_offset_start: usize,
    /// Offset of scales end from the end of the buffer
    pub scales_offset_end: usize,
    /// Shape of the scales tensor
    pub scales_shape: Shape,
    /// Strides of the scales tensor
    pub scales_strides: Vec<usize>,
    /// Type of the scales
    pub scales_dtype: DType,
}

impl<R: CubeRuntime> CubeTensor<R> {
    /// Construct a separate tensor for the quantization scales, if present
    pub fn scales(&self) -> Option<CubeTensor<R>> {
        let qparams = self.qparams.as_ref()?;
        let mut handle = self.handle.clone();
        handle.offset_start = Some(qparams.scales_offset_start as u64);
        handle.offset_end = Some(qparams.scales_offset_end as u64);

        Some(CubeTensor::new(
            self.client.clone(),
            handle,
            qparams.scales_shape.clone(),
            self.device.clone(),
            qparams.scales_strides.clone(),
            qparams.scales_dtype,
        ))
    }
}
