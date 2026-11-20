use crate::{LearningRate, Optimizer, OptimizerStateMapper};
use burn_core::tensor::Tensor;

/// An optimizer whose floating-point statistics can be transformed on device.
/// Its state record must reserve the top-level name `__precision` for the wrapper.
/// Derive [`crate::RecordState`] for records with tensor leaves, including nested
/// states, or use [`crate::StateSource::tensor_options`] when restoring those leaves by hand.
pub trait ConfigurableOptimizer: Optimizer {
    /// Apply `map` to every statistics tensor, preserving its shape and scalar bookkeeping.
    fn map_state<const D: usize, M: OptimizerStateMapper>(
        state: Self::State<D>,
        mapper: &mut M,
    ) -> Self::State<D>;

    /// Apply the equation in the supplied tensors' dtype. Override when the ordinary
    /// step selects a different dtype internally.
    fn step_in_dtype<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        self.step(lr, tensor, grad, state)
    }
}
