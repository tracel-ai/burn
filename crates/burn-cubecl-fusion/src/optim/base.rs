use crate::{CubeFusionHandle, FallbackOperation};
use burn_fusion::stream::Context;
use cubecl::Runtime;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// The runnable result of fusing a segment of tensor operations — the single
/// trait the built-in fusion optimizations and user-defined ones implement
/// alike. A fuser's [`finish`](burn_fusion::OperationFuser::finish) wraps its
/// fused operation in a [`CubeOptimization`], the type the fusion runtime
/// executes.
///
/// User-defined fused operations are registered through the optimization
/// registry in `burn-cubecl` (`fusion::register`).
pub trait FusedOperation<R: Runtime>: Send + 'static {
    /// Name of the fused operation — the key serialized execution plans are
    /// restored by, also shown in diagnostics and fusion logs.
    const NAME: &'static str;

    /// The serializable state of the fused operation.
    type State: Serialize + DeserializeOwned;

    /// The number of operations fused.
    fn num_ops_fused(&self) -> usize;

    /// Run the fused operation. `fallback` builds the unfused operation at
    /// the given index within the segment, for implementations that need to
    /// run part of it unfused (autotune fallbacks).
    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    );

    /// The state of the fused operation, from which
    /// [`from_state`](Self::from_state) recovers it.
    fn to_state(&self) -> Self::State;

    /// Recover the fused operation from its [state](Self::to_state).
    fn from_state(device: &R::Device, state: Self::State) -> Self;
}

/// A fusion optimization ready to run on an execution stream: what fusers
/// finish and the fusion runtime executes. Wraps any [`FusedOperation`].
pub struct CubeOptimization<R: Runtime> {
    optimization: Box<dyn DynFusedOperation<R>>,
}

impl<R: Runtime> CubeOptimization<R> {
    /// Wrap the fused operation.
    pub fn new(optimization: impl FusedOperation<R>) -> Self {
        Self {
            optimization: Box::new(optimization),
        }
    }

    /// The optimization's [name](FusedOperation::NAME).
    pub fn name(&self) -> &'static str {
        self.optimization.name()
    }

    /// The number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.optimization.num_ops_fused()
    }

    /// Run the fused operation. See [`FusedOperation::run`].
    pub fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        self.optimization.run(context, fallback)
    }

    /// Serialize the optimization: its name plus its
    /// [state](FusedOperation::to_state) encoded as bytes.
    pub fn to_state(&self) -> CubeOptimizationState {
        self.optimization.to_state()
    }
}

impl<R: Runtime> core::fmt::Debug for CubeOptimization<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} ({} ops)", self.name(), self.num_ops_fused())
    }
}

impl<R: Runtime> burn_fusion::NumOperations for CubeOptimization<R> {
    fn len(&self) -> usize {
        self.num_ops_fused()
    }

    fn name(&self) -> &'static str {
        Self::name(self)
    }
}

/// Object-safe view of a [`FusedOperation`], implemented for every one of
/// them below. Private on purpose: the erasure, like the box holding it, is an
/// implementation detail of [`CubeOptimization`].
trait DynFusedOperation<R: Runtime>: Send {
    fn name(&self) -> &'static str;
    fn num_ops_fused(&self) -> usize;
    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    );
    fn to_state(&self) -> CubeOptimizationState;
}

impl<R: Runtime, O: FusedOperation<R>> DynFusedOperation<R> for O {
    fn name(&self) -> &'static str {
        O::NAME
    }

    fn num_ops_fused(&self) -> usize {
        FusedOperation::num_ops_fused(self)
    }

    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        FusedOperation::run(self, context, fallback)
    }

    fn to_state(&self) -> CubeOptimizationState {
        CubeOptimizationState::new(O::NAME, &FusedOperation::to_state(self))
    }
}

/// Serializable state of a fusion optimization: its name plus its own state
/// encoded as bytes, so one type covers built-in and user-defined
/// optimizations alike.
#[derive(Serialize, Deserialize, Debug)]
pub struct CubeOptimizationState {
    /// The optimization's [name](FusedOperation::NAME) — the key
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
