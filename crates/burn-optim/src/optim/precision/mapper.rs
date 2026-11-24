use crate::StatePath;
use burn_core::tensor::Tensor;

/// Transforms optimizer state leaves on device without changing their shape or device.
/// Paths identify leaves by their complete state-record field path.
/// Implementations must handle every tensor rank and must not change scalar bookkeeping.
pub trait OptimizerStateMapper {
    /// Transform one tensor, preserving its shape and device.
    fn map<const D: usize>(&self, tensor: Tensor<D>, path: &StatePath<'_>) -> Tensor<D>;
}
