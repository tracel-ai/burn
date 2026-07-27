use crate::optim::{
    elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
    matmul::{MatmulOptimization, MatmulOptimizationState},
    reduce::{ReduceOptimization, ReduceOptimizationState},
    reduce_broadcasted::{ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationState},
};
use crate::{CubeFusionHandle, FallbackOperation};
use burn_fusion::stream::Context;
use cubecl::Runtime;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A fusion optimization for cubecl backends — the single trait the built-in
/// optimizations and user-defined ones implement alike. A fuser's
/// [`finish`](burn_fusion::OperationFuser::finish) wraps its optimization in a
/// [`CubeOptim`], the type the fusion runtime executes.
///
/// User-defined optimizations are registered through the optimization registry
/// in `burn-cubecl` (`fusion::register`).
pub trait CubeOptimization<R: Runtime>: Send + 'static {
    /// Name of the optimization — the key serialized execution plans are
    /// restored by, also shown in diagnostics and fusion logs.
    const NAME: &'static str;

    /// The serializable state of the optimization.
    type State: Serialize + DeserializeOwned;

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

    /// The state of the optimization, from which [`from_state`](Self::from_state)
    /// recovers it.
    fn to_state(&self) -> Self::State;

    /// Recover the optimization from its [state](Self::to_state).
    fn from_state(device: &R::Device, state: Self::State) -> Self;
}

/// A fusion optimization ready to run on an execution stream: what fusers
/// finish and the fusion runtime executes. Wraps any [`CubeOptimization`].
pub struct CubeOptim<R: Runtime> {
    optimization: Box<dyn DynOptimization<R>>,
}

impl<R: Runtime> CubeOptim<R> {
    /// Wrap the optimization.
    pub fn new(optimization: impl CubeOptimization<R>) -> Self {
        Self {
            optimization: Box::new(optimization),
        }
    }

    /// The optimization's [name](CubeOptimization::NAME).
    pub fn name(&self) -> &'static str {
        self.optimization.name()
    }

    /// The number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.optimization.num_ops_fused()
    }

    /// Execute the optimization. See [`CubeOptimization::execute`].
    pub fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        self.optimization.execute(context, fallback)
    }

    /// Serialize the optimization: its name plus its
    /// [state](CubeOptimization::to_state) encoded as bytes.
    pub fn to_state(&self) -> CubeOptimizationState {
        self.optimization.to_state()
    }
}

impl<R: Runtime> core::fmt::Debug for CubeOptim<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} ({} ops)", self.name(), self.num_ops_fused())
    }
}

impl<R: Runtime> burn_fusion::NumOperations for CubeOptim<R> {
    fn len(&self) -> usize {
        self.num_ops_fused()
    }

    fn name(&self) -> &'static str {
        Self::name(self)
    }
}

/// Object-safe view of a [`CubeOptimization`], implemented for every one of
/// them below. Private on purpose: the erasure, like the box holding it, is an
/// implementation detail of [`CubeOptim`].
trait DynOptimization<R: Runtime>: Send {
    fn name(&self) -> &'static str;
    fn num_ops_fused(&self) -> usize;
    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    );
    fn to_state(&self) -> CubeOptimizationState;
}

impl<R: Runtime, O: CubeOptimization<R>> DynOptimization<R> for O {
    fn name(&self) -> &'static str {
        O::NAME
    }

    fn num_ops_fused(&self) -> usize {
        CubeOptimization::num_ops_fused(self)
    }

    fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        CubeOptimization::execute(self, context, fallback)
    }

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(O::NAME, &CubeOptimization::to_state(self))
    }
}

/// Serializable state of a fusion optimization: its name plus its own state
/// encoded as bytes, so one type covers built-in and user-defined
/// optimizations alike.
#[derive(Serialize, Deserialize, Debug)]
pub struct CubeOptimizationState {
    /// The optimization's [name](CubeOptimization::NAME) — the key
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
    pub fn decode<T: DeserializeOwned>(&self) -> T {
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

/// Names of the built-in fusion optimizations, matching each one's
/// [`CubeOptimization::NAME`].
pub const BUILTIN_NAMES: [&str; 4] = [ELEMWISE, MATMUL, REDUCE, REDUCE_BROADCASTED];

/// Restore a built-in optimization from its [state](CubeOptimizationState),
/// or `None` when the name is not a built-in's.
pub fn restore_builtin<R: Runtime>(
    device: &R::Device,
    state: &CubeOptimizationState,
) -> Option<CubeOptim<R>> {
    Some(match state.name.as_str() {
        ELEMWISE => CubeOptim::new(ElemwiseOptimization::<R>::from_state(device, state.decode())),
        MATMUL => CubeOptim::new(MatmulOptimization::<R>::from_state(device, state.decode())),
        REDUCE => CubeOptim::new(ReduceOptimization::<R>::from_state(device, state.decode())),
        REDUCE_BROADCASTED => CubeOptim::new(ReduceBroadcastedOptimization::<R>::from_state(
            device,
            state.decode(),
        )),
        _ => return None,
    })
}

impl<R: Runtime> CubeOptimization<R> for ElemwiseOptimization<R> {
    const NAME: &'static str = ELEMWISE;
    type State = ElemwiseOptimizationState;

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

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &R::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}

impl<R: Runtime> CubeOptimization<R> for MatmulOptimization<R> {
    const NAME: &'static str = MATMUL;
    type State = MatmulOptimizationState;

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

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &R::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceOptimization<R> {
    const NAME: &'static str = REDUCE;
    type State = ReduceOptimizationState;

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

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &R::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}

impl<R: Runtime> CubeOptimization<R> for ReduceBroadcastedOptimization<R> {
    const NAME: &'static str = REDUCE_BROADCASTED;
    type State = ReduceBroadcastedOptimizationState;

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

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &R::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
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
