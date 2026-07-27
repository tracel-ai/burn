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
    /// Name of the optimization — the key serialized execution plans are
    /// restored by, also shown in diagnostics and fusion logs.
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

    /// The serializable state of the optimization, built with
    /// [`CubeOptimizationState::new`] under [`Self::name`]. Restoring reverses
    /// it: the state is dispatched by name to the matching built-in or
    /// registered provider, which [decodes](CubeOptimizationState::decode) it.
    fn to_state(&self) -> CubeOptimizationState;
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

/// Serializable state of a fusion optimization: its name plus its own state
/// encoded as bytes, so one type covers built-in and user-defined
/// optimizations alike.
#[derive(Serialize, Deserialize, Debug)]
pub struct CubeOptimizationState {
    /// The optimization's [name](CubeOptimization::name) — the key
    /// restoration dispatches on.
    pub name: String,
    state: Vec<u8>,
}

impl CubeOptimizationState {
    /// Encode the state of the optimization named `name`.
    pub fn new(name: &str, state: &impl Serialize) -> Self {
        Self {
            name: name.to_string(),
            state: rmp_serde::to_vec(state)
                .expect("fusion optimization state must be serializable"),
        }
    }

    /// Decode the optimization's state.
    ///
    /// # Panics
    ///
    /// When the bytes don't decode as `T` — the state was produced by a
    /// different optimization sharing the name, or by an incompatible version.
    pub fn decode<T: serde::de::DeserializeOwned>(&self) -> T {
        rmp_serde::from_slice(&self.state).unwrap_or_else(|err| {
            panic!(
                "state of fusion optimization `{}` failed to decode: {err}",
                self.name
            )
        })
    }
}

const ELEMWISE: &str = "ElementWise";
const MATMUL: &str = "Matmul";
const REDUCE: &str = "Reduce";
const REDUCE_BROADCASTED: &str = "ReduceBroadcasted";

/// Names of the built-in fusion optimizations, matching what
/// [`CubeOptimization::name`] returns for each.
pub const BUILTIN_NAMES: [&str; 4] = [ELEMWISE, MATMUL, REDUCE, REDUCE_BROADCASTED];

/// Restore a built-in optimization from its [state](CubeOptimizationState),
/// or `None` when the name is not a built-in's.
pub fn restore_builtin<R: Runtime>(
    device: &R::Device,
    state: &CubeOptimizationState,
) -> Option<Box<dyn CubeOptimization<R>>> {
    Some(match state.name.as_str() {
        ELEMWISE => Box::new(ElemwiseOptimization::<R>::from_state(device, state.decode())),
        MATMUL => Box::new(MatmulOptimization::<R>::from_state(device, state.decode())),
        REDUCE => Box::new(ReduceOptimization::<R>::from_state(device, state.decode())),
        REDUCE_BROADCASTED => Box::new(ReduceBroadcastedOptimization::<R>::from_state(
            device,
            state.decode(),
        )),
        _ => return None,
    })
}

impl<R: Runtime> CubeOptimization<R> for ElemwiseOptimization<R> {
    fn name(&self) -> &'static str {
        ELEMWISE
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

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(ELEMWISE, &Self::to_state(self))
    }
}

impl<R: Runtime> CubeOptimization<R> for MatmulOptimization<R> {
    fn name(&self) -> &'static str {
        MATMUL
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

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(MATMUL, &Self::to_state(self))
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceOptimization<R> {
    fn name(&self) -> &'static str {
        REDUCE
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

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(REDUCE, &Self::to_state(self))
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceBroadcastedOptimization<R> {
    fn name(&self) -> &'static str {
        REDUCE_BROADCASTED
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

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(REDUCE_BROADCASTED, &Self::to_state(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_through_bytes() {
        let state = CubeOptimizationState::new("custom", &vec![3u32, 7]);

        assert_eq!(state.name, "custom");
        assert_eq!(state.decode::<Vec<u32>>(), vec![3, 7]);
    }
}
