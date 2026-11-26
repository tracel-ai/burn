use crate::{Optimizer, OptimizerStateMapper};

/// An optimizer whose floating-point statistics can be transformed on device.
/// Implement this trait to use uniform dtype storage and per-tensor storage mappers.
/// Derive [`crate::StateMappable`] for the statistics and delegate to its
/// [`map_tensors`](crate::StateMappable::map_tensors) method.
pub trait MappableOptimizer: Optimizer {
    /// Apply `map` to every statistics tensor, preserving its shape and scalar bookkeeping.
    fn map_state<const D: usize, M: OptimizerStateMapper>(
        state: Self::State<D>,
        mapper: &M,
    ) -> Self::State<D>;
}

// Bridge the rank-generic optimizer state to the traversal derived on each concrete record.
macro_rules! mappable_optimizers {
    ($($optimizer:ident),* $(,)?) => {$(
        impl MappableOptimizer for crate::$optimizer {
            fn map_state<const D: usize, M: OptimizerStateMapper>(
                state: Self::State<D>, mapper: &M,
            ) -> Self::State<D> {
                crate::StateMappable::map_tensors(state, mapper)
            }
        }
    )*};
}

mappable_optimizers!(
    Adafactor, AdaGrad, Adam, AdamW, Adan, Lamb, Lion, Muon, RmsProp, Sgd
);
