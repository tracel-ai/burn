use crate::StatePath;
use burn_core::tensor::{Bool, Int, Tensor};

/// Visits tensor leaves by record path, preserving their shapes.
/// A device migration can move leaves. A mapper used for statistics storage must
/// keep them on the update device and preserve integer and boolean bookkeeping.
pub trait OptimizerStateMapper {
    /// Transform a floating-point statistic.
    fn map<const D: usize>(&self, tensor: Tensor<D>, path: &StatePath<'_>) -> Tensor<D>;

    /// Visit integer bookkeeping. Storage mappers can use the identity default.
    fn map_int<const D: usize>(
        &self,
        tensor: Tensor<D, Int>,
        _path: &StatePath<'_>,
    ) -> Tensor<D, Int> {
        tensor
    }

    /// Visit boolean bookkeeping. Storage mappers can use the identity default.
    fn map_bool<const D: usize>(
        &self,
        tensor: Tensor<D, Bool>,
        _path: &StatePath<'_>,
    ) -> Tensor<D, Bool> {
        tensor
    }
}
