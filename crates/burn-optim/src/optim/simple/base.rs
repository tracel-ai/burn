use std::any::Any;
use std::sync::Arc;

use burn_core as burn;
use burn_core::tensor::kind::BridgeTensor;

use crate::LearningRate;
use burn::record::Record;
use burn::tensor::{Device, Tensor};

/// Simple optimizer is an opinionated trait to simplify the process of implementing an
/// optimizer.
///
/// Implementations don't have to handle missing gradients, loading and exporting records, navigate the
/// module parameter structure, handle tracked and untracked tensors, and the likes.
pub trait OptimizerStep: Send + Sync + Clone + 'static {
    /// The state of the optimizer. It also implements [record](Record), so that it can be saved.
    type State<const D: usize>: Send + Sync + Record + Clone + 'static;

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
    /// gradient and the tensor when the [step](SimpleOptimizer::step) function is called.
    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D>;
}

#[derive(Clone)]
pub struct DynState {
    state: Arc<dyn Any + Send + Sync>,
}

impl DynState {
    pub fn downcast<T: Clone + 'static>(self) -> T {
        todo!()
    }

    pub fn create<T: Send + Sync + 'static>(state: T) -> Self {
        Self {
            state: Arc::new(state),
        }
    }
}

pub(crate) trait DynOptimizer: Send + Sync {
    fn step_dyn(
        &self,
        rank: usize,
        lr: LearningRate,
        tensor: BridgeTensor,
        grad: BridgeTensor,
        state: Option<DynState>,
    ) -> (BridgeTensor, Option<DynState>);
    fn to_device_dyn(&self, state: DynState, device: &Device) -> DynState;
}

impl<O: OptimizerStep> DynOptimizer for O {
    fn step_dyn(
        &self,
        rank: usize,
        lr: LearningRate,
        tensor: BridgeTensor,
        grad: BridgeTensor,
        state: Option<DynState>,
    ) -> (BridgeTensor, Option<DynState>) {
        match rank {
            1 => {
                let (grad, state) = O::step(
                    &self,
                    lr,
                    Tensor::<1>::from_bridge(tensor),
                    Tensor::<1>::from_bridge(grad),
                    state.map(|s| s.downcast()),
                );

                (grad.into_bridge(), state.map(|s| DynState::create(s)))
            }
            _ => panic!("Unsupported rank"),
        }
    }

    fn to_device_dyn(&self, state: DynState, device: &Device) -> DynState {
        todo!()
    }
}
