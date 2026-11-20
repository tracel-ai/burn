use crate::{OptimizerStateMapper, RecordState};

/// State whose floating tensors can be transformed, including nested and optional components.
/// Implementations visit every tensor exactly once and preserve scalar bookkeeping.
pub trait StateMappable: RecordState + Clone {
    /// Apply a mapper to each tensor leaf while preserving the structure of the state.
    fn map_tensors<M: OptimizerStateMapper>(self, mapper: &mut M) -> Self;
}

impl StateMappable for () {
    fn map_tensors<M: OptimizerStateMapper>(self, _mapper: &mut M) -> Self {}
}
