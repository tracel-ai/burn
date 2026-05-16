use burn_core as burn;

use crate::LearningRate;
use burn::record::Record;
use burn::tensor::{Device, Tensor};

/// Simple optimizer is an opinionated trait to simplify the process of implementing an
/// optimizer.
///
/// Implementations don't have to handle missing gradients, loading and exporting records, navigate the
/// module parameter structure, handle tracked and untracked tensors, and the likes.
pub trait SimpleOptimizer: Send + Sync + Clone {
    /// The state of the optimizer. It also implements [record](Record), so that it can be saved.
    type State<const D: usize>: Record + Clone + 'static;

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
