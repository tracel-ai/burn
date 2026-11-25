use crate::{MappableOptimizer, Optimizer, OptimizerStateMapper, RecordState, StatePath};
use burn_core::tensor::{Device, FloatDType, Tensor};

/// Representation of one parameter's optimizer statistics between updates.
///
/// The stored record owns all data needed to decode it, including per-block scales or other
/// auxiliary tensors. Its record schema determines checkpoint compatibility. Use the same
/// storage implementation to resume a checkpoint with a custom schema. Reserve the
/// top-level record name `__precision` for optional master weights. Use [`RecordState`]'s
/// derive or [`crate::StateSource::read_tensor`] to restore leaves in their stored dtypes.
pub trait OptimizerStateStorage<O: Optimizer>: Clone + Send + Sync + 'static {
    /// Checkpointable representation for a parameter of rank `D`.
    type Stored<const D: usize>: Clone + RecordState;

    /// Encode statistics on the update device after the equation produces the parameter update.
    fn store<const D: usize>(&self, state: O::State<D>) -> Self::Stored<D>;

    /// Reconstruct the statistics for the next update, in its compute dtype.
    fn restore<const D: usize>(&self, stored: Self::Stored<D>, compute: FloatDType) -> O::State<D>;

    /// Move the entire encoded record, including auxiliary data, to the update device.
    fn to_device<const D: usize>(stored: Self::Stored<D>, device: &Device) -> Self::Stored<D>;
}

/// A tensor mapper suffices when each stored leaf can be decoded by casting the tensor.
/// The original record names, counters and optional components are preserved.
impl<O, M> OptimizerStateStorage<O> for M
where
    O: MappableOptimizer,
    M: OptimizerStateMapper + Clone + Send + Sync + 'static,
{
    type Stored<const D: usize> = O::State<D>;

    fn store<const D: usize>(&self, state: O::State<D>) -> Self::Stored<D> {
        O::map_state(state, self)
    }

    fn restore<const D: usize>(&self, stored: Self::Stored<D>, compute: FloatDType) -> O::State<D> {
        O::map_state(stored, &compute)
    }

    fn to_device<const D: usize>(stored: Self::Stored<D>, device: &Device) -> Self::Stored<D> {
        O::to_device::<D>(stored, device)
    }
}

impl OptimizerStateMapper for FloatDType {
    fn map<const D: usize>(&self, tensor: Tensor<D>, _: &StatePath<'_>) -> Tensor<D> {
        tensor.cast(*self)
    }
}
