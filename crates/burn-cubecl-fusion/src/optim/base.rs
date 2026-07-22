use crate::optim::{
    elemwise::ElemwiseOptimization, matmul::MatmulOptimization, reduce::ReduceOptimization,
    reduce_broadcasted::ReduceBroadcastedOptimization,
};
use crate::{CubeFusionHandle, FallbackOperation};
use burn_fusion::stream::Context;
use cubecl::Runtime;
use serde::{Deserialize, Serialize};

/// A fusion optimization for cubecl backends — the single trait the built-in
/// optimizations and user-defined ones implement alike. The runtime's
/// optimization type is `Box<dyn CubeOptimization<R>>`; a fuser's
/// [`finish`](burn_fusion::OperationFuser::finish) boxes its optimization.
///
/// User-defined optimizations are registered through the optimization registry
/// in `burn-cubecl` (`fusion::register`).
pub trait CubeOptimization<R: Runtime>: Send {
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
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    );
}

impl<R: Runtime> core::fmt::Debug for dyn CubeOptimization<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} ({} ops)", self.name(), self.num_ops_fused())
    }
}

impl<R: Runtime> burn_fusion::NumOperations for Box<dyn CubeOptimization<R>> {
    fn len(&self) -> usize {
        self.as_ref().num_ops_fused()
    }

    fn name(&self) -> &'static str {
        self.as_ref().name()
    }
}

/// Serializable stand-in for a fusion optimization: its name only. Restoring
/// an optimization from state is not supported by the cubecl fusion runtime —
/// optimizations are rebuilt by their fusers, never deserialized.
#[derive(Serialize, Deserialize, Debug)]
pub struct CubeOptimizationState {
    /// The optimization's [name](CubeOptimization::name).
    pub name: String,
}

impl<R: Runtime> CubeOptimization<R> for ElemwiseOptimization<R> {
    fn name(&self) -> &'static str {
        "ElementWise"
    }

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        _fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        Self::execute(self, context)
    }
}

impl<R: Runtime> CubeOptimization<R> for MatmulOptimization<R> {
    fn name(&self) -> &'static str {
        "Matmul"
    }

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        Self::execute(self, context, |index| fallback(index))
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceOptimization<R> {
    fn name(&self) -> &'static str {
        "Reduce"
    }

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        Self::execute(self, context, |index| fallback(index))
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceBroadcastedOptimization<R> {
    fn name(&self) -> &'static str {
        "ReduceBroadcasted"
    }

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        Self::execute(self, context, |index| fallback(index))
    }
}
