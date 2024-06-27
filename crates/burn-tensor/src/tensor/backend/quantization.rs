use crate::ops::QTensorOps;

use super::Backend;

/// This trait defines the type and functions needed for a quantization backend.
/// It is an extension of the [`Backend`] trait.
pub trait QuantizationBackend: Backend + QTensorOps<Self> {
    /// Tensor primitive to be used for all quantized operations.
    type QuantizedTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;
}
