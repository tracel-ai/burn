use crate::optim::{
    elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
    matmul::{MatmulOptimization, MatmulOptimizationState},
    reduce::{ReduceOptimization, ReduceOptimizationState},
    reduce_broadcasted::{ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationState},
};
use cubecl::Runtime;
use serde::{Deserialize, Serialize};

/// Fusion optimization type for cubecl.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
pub enum CubeOptimization<R: Runtime> {
    ElementWise(ElemwiseOptimization<R>),
    Matmul(MatmulOptimization<R>),
    Reduce(ReduceOptimization<R>),
    ReduceBroadcasted(ReduceBroadcastedOptimization<R>),
}

impl<R: Runtime> core::fmt::Debug for CubeOptimization<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.to_opt_state();
        f.write_fmt(format_args!("{value:?}"))
    }
}

impl<R: Runtime> CubeOptimization<R> {
    /// Serializes the current optimization to its state.
    pub fn to_opt_state(&self) -> CubeOptimizationState {
        match self {
            Self::ElementWise(value) => CubeOptimizationState::ElementWise(value.to_state()),
            Self::Matmul(value) => CubeOptimizationState::Matmul(value.to_state()),
            Self::Reduce(value) => CubeOptimizationState::Reduce(value.to_state()),
            Self::ReduceBroadcasted(value) => {
                CubeOptimizationState::ReduceBroadcasted(value.to_state())
            }
        }
    }
}

impl<R: Runtime> burn_fusion::NumOperations for CubeOptimization<R> {
    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.num_ops_fused(),
            Self::Matmul(op) => op.num_ops_fused(),
            Self::Reduce(op) => op.num_ops_fused(),
            Self::ReduceBroadcasted(op) => op.num_ops_fused(),
        }
    }
}

/// Fusion optimization state type for cubecl.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum CubeOptimizationState {
    ElementWise(ElemwiseOptimizationState),
    Matmul(MatmulOptimizationState),
    Reduce(ReduceOptimizationState),
    ReduceBroadcasted(ReduceBroadcastedOptimizationState),
}
