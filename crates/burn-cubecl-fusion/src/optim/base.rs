use crate::optim::{
    elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
    matmul::{MatmulOptimization, MatmulOptimizationState},
    reduce::{ReduceOptimization, ReduceOptimizationState},
    reduce_broadcasted::{ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationState},
};
use crate::{CubeFusionHandle, FallbackOperation};
use burn_fusion::stream::Context;
use cubecl::Runtime;
use serde::{Deserialize, Serialize};

/// A user-defined fusion optimization, held by [`CubeOptimization::Custom`].
///
/// Built by the [`OperationFuser`](burn_fusion::OperationFuser) of a provider
/// registered through [`burn_fusion::register`]: the fuser's `finish` wraps an
/// implementation of this trait in [`CubeOptimization::Custom`].
pub trait CustomOptimization<R: Runtime>: Send {
    /// Name of the optimization, for diagnostics and fusion logs.
    fn name(&self) -> &'static str;

    /// The number of operations fused.
    fn num_ops_fused(&self) -> usize;

    /// Execute the optimization. `fallback` builds the unfused operation at
    /// the given index within the optimization, for implementations that need
    /// to run part of the segment unfused (autotune fallbacks).
    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &mut dyn FnMut(usize) -> Box<dyn FallbackOperation<R>>,
    );
}

/// Fusion optimization type for cubecl.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
pub enum CubeOptimization<R: Runtime> {
    ElementWise(ElemwiseOptimization<R>),
    Matmul(MatmulOptimization<R>),
    Reduce(ReduceOptimization<R>),
    ReduceBroadcasted(ReduceBroadcastedOptimization<R>),
    /// A user-defined optimization (see [`CustomOptimization`]).
    Custom(Box<dyn CustomOptimization<R>>),
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
            Self::Custom(value) => CubeOptimizationState::Custom {
                name: value.name().to_string(),
            },
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
            Self::Custom(op) => op.num_ops_fused(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CubeOptimization::ElementWise(..) => "ElementWise",
            CubeOptimization::Matmul(..) => "Matmul",
            CubeOptimization::Reduce(..) => "Reduce",
            CubeOptimization::ReduceBroadcasted(..) => "ReduceBroadcasted",
            CubeOptimization::Custom(op) => op.name(),
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
    /// A user-defined optimization carries no restorable state — only its
    /// name, for diagnostics. See [`CustomOptimization`].
    Custom {
        /// The optimization's [name](CustomOptimization::name).
        name: String,
    },
}
