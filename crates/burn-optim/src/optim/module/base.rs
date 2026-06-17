use alloc::sync::Arc;
use core::any::Any;

use crate::{RecordState, StateSink, StateSource};
use burn_core as burn;
use burn_core::tensor::kind::BridgeTensor;

use crate::LearningRate;
use burn::tensor::{Device, Tensor};

/// An opinionated trait to simplify the process of implementing an optimizer.
///
/// Implementations don't have to handle missing gradients, loading and exporting records,
/// navigate the module parameter structure, handle tracked and untracked tensors, and the likes.
/// Wrap one in a [`ModuleOptimizer`](crate::optim::ModuleOptimizer) to optimize a whole module.
pub trait Optimizer: Send + Sync + Clone + 'static {
    /// The state of the optimizer for a single parameter of rank `D`.
    ///
    /// It implements [`RecordState`] (which itself requires `Send + Sync + 'static`) so it can be
    /// decomposed into named tensors and scalars for the burnpack format.
    type State<const D: usize>: Clone + RecordState;

    /// The optimizer step is performed for one tensor at a time with its gradient and state.
    ///
    /// Note that the state is passed as parameter, so implementations don't have to handle
    /// the saving and loading of recorded states.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>);

    /// Change the device of the state.
    ///
    /// This function will be called accordingly to have the state on the same device as the
    /// gradient and the tensor when the [step](Optimizer::step) function is called.
    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D>;
}

/// A type-erased optimizer state for a single parameter.
///
/// It wraps a concrete `O::State<D>` together with its rank `D` so that the rank can be recovered
/// when the state is later interpreted by the originating [`Optimizer`] (during a step, a device
/// transfer or serialization).
#[derive(Clone)]
pub(crate) struct DynState {
    state: Arc<dyn Any + Send + Sync>,
    rank: usize,
}

impl DynState {
    /// Erase a concrete optimizer state of rank `rank`.
    pub fn create<T: Send + Sync + 'static>(state: T, rank: usize) -> Self {
        Self {
            state: Arc::new(state),
            rank,
        }
    }

    /// Recover the concrete state by value, moving it out when this is the only handle and
    /// cloning only when the underlying state is still shared. Panics if `T` does not match the
    /// stored type.
    pub fn downcast<T: Clone + Send + Sync + 'static>(self) -> T {
        let state = self
            .state
            .downcast::<T>()
            .expect("The dynamic optimizer state should match the optimizer state type.");
        Arc::try_unwrap(state).unwrap_or_else(|state| (*state).clone())
    }

    /// Borrow the concrete state without cloning. Panics if `T` does not match the stored type.
    pub fn downcast_ref<T: 'static>(&self) -> &T {
        self.state
            .downcast_ref::<T>()
            .expect("The dynamic optimizer state should match the optimizer state type.")
    }

    /// The rank of the parameter this state belongs to.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Dispatch a runtime `rank` to a body parameterized by a `const D: usize`.
macro_rules! dispatch_rank {
    ($rank:expr, $d:ident => $body:block) => {
        match $rank {
            0 => {
                const $d: usize = 0;
                $body
            }
            1 => {
                const $d: usize = 1;
                $body
            }
            2 => {
                const $d: usize = 2;
                $body
            }
            3 => {
                const $d: usize = 3;
                $body
            }
            4 => {
                const $d: usize = 4;
                $body
            }
            5 => {
                const $d: usize = 5;
                $body
            }
            6 => {
                const $d: usize = 6;
                $body
            }
            7 => {
                const $d: usize = 7;
                $body
            }
            8 => {
                const $d: usize = 8;
                $body
            }
            other => panic!("Unsupported tensor rank for optimizer state: {other}"),
        }
    };
}

/// Object-safe view over an [`Optimizer`], allowing [`ModuleOptimizer`](crate::optim::ModuleOptimizer)
/// to stay non-generic. Rank-generic operations are dispatched on a runtime rank.
pub(crate) trait DynOptimizer: Send + Sync {
    /// Perform an optimizer step for a single parameter of the given `rank`.
    fn step_dyn(
        &self,
        rank: usize,
        lr: LearningRate,
        tensor: BridgeTensor,
        grad: BridgeTensor,
        state: Option<DynState>,
    ) -> (BridgeTensor, Option<DynState>);

    /// Move a state to the given device.
    fn to_device_dyn(&self, state: DynState, device: &Device) -> DynState;

    /// Decompose a state into named tensors and scalars under `prefix`.
    fn state_flatten(&self, prefix: &str, state: &DynState, out: &mut StateSink);

    /// Rebuild a state of the given `rank` from named tensors and scalars under `prefix`.
    ///
    /// Returns `None` when the record does not contain a reconstructable state for this parameter
    /// (e.g. a truncated or foreign file); the caller leaves that parameter without state, so it is
    /// re-initialized on the next step.
    fn state_unflatten(
        &self,
        rank: usize,
        prefix: &str,
        src: &mut StateSource,
        device: &Device,
    ) -> Option<DynState>;
}

impl<O: Optimizer> DynOptimizer for O {
    fn step_dyn(
        &self,
        rank: usize,
        lr: LearningRate,
        tensor: BridgeTensor,
        grad: BridgeTensor,
        state: Option<DynState>,
    ) -> (BridgeTensor, Option<DynState>) {
        dispatch_rank!(rank, D => {
            let (tensor, state) = self.step(
                lr,
                Tensor::<D>::from_bridge(tensor),
                Tensor::<D>::from_bridge(grad),
                state.map(|state| state.downcast::<O::State<D>>()),
            );

            (tensor.into_bridge(), state.map(|state| DynState::create(state, D)))
        })
    }

    fn to_device_dyn(&self, state: DynState, device: &Device) -> DynState {
        dispatch_rank!(state.rank(), D => {
            let state = O::to_device::<D>(state.downcast::<O::State<D>>(), device);
            DynState::create(state, D)
        })
    }

    fn state_flatten(&self, prefix: &str, state: &DynState, out: &mut StateSink) {
        dispatch_rank!(state.rank(), D => {
            RecordState::state_flatten(state.downcast_ref::<O::State<D>>(), prefix, out);
        })
    }

    fn state_unflatten(
        &self,
        rank: usize,
        prefix: &str,
        src: &mut StateSource,
        device: &Device,
    ) -> Option<DynState> {
        dispatch_rank!(rank, D => {
            let state = <O::State<D> as RecordState>::state_unflatten(prefix, src, device)?;
            Some(DynState::create(state, D))
        })
    }
}
