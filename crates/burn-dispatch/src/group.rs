#[cfg(feature = "autodiff")]
use alloc::boxed::Box;

#[cfg(feature = "autodiff")]
use burn_autodiff::{
    checkpoint::{
        base::Checkpointer,
        strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing},
    },
    grads::Gradients,
    ops::{Backward, Ops, unary},
};
#[cfg(feature = "autodiff")]
use burn_backend::{Backend, tensor::FloatTensor};
use burn_group::GroupPlacement;

use crate::{BackendTensor, DispatchTensor, DispatchTensorKind, backends::Group};
#[cfg(feature = "autodiff")]
use crate::{DispatchAutodiffContext, GradientCheckpointingStrategy, backends::Autodiff};

impl DispatchTensor {
    /// The same value at another placement over its device group.
    ///
    /// A tracked autodiff tensor records the move: its gradient flows back through it unchanged,
    /// since a gradient may sit at any placement.
    ///
    /// # Panics
    ///
    /// When the tensor is not on a device group, or no collective produces `placement`: a
    /// partial sum out of a whole tensor, or a split along a dim shorter than the group.
    pub fn place_in_group(self, placement: GroupPlacement) -> Self {
        let DispatchTensor { kind, autodiff } = self;
        let kind = match kind {
            DispatchTensorKind::Group(tensor) => {
                DispatchTensorKind::Group(tensor.place_in_group(placement))
            }
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner) => {
                let DispatchAutodiffContext::Enabled(checkpointing) = autodiff else {
                    unreachable!("An autodiff tensor has an enabled autodiff context")
                };
                let DispatchTensorKind::Group(BackendTensor::Autodiff(tensor)) = *inner else {
                    panic!("Only a tensor on a device group has a placement")
                };
                let placed = match checkpointing {
                    GradientCheckpointingStrategy::Balanced => {
                        PlaceInGroup::record::<BalancedCheckpointing>(tensor, placement)
                    }
                    GradientCheckpointingStrategy::Disabled => {
                        PlaceInGroup::record::<NoCheckpointing>(tensor, placement)
                    }
                };
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Group(
                    BackendTensor::Autodiff(placed),
                )))
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Only a tensor on a device group has a placement"),
        };
        DispatchTensor { kind, autodiff }
    }

    /// Where the tensor's shards sit, or `None` when it is not on a device group.
    pub fn group_placement(&self) -> Option<GroupPlacement> {
        let primitive = match &self.kind {
            DispatchTensorKind::Group(
                BackendTensor::Float(tensor)
                | BackendTensor::Int(tensor)
                | BackendTensor::Bool(tensor),
            ) => tensor,
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner) => match &**inner {
                DispatchTensorKind::Group(BackendTensor::Autodiff(tensor)) => tensor.primitive(),
                _ => return None,
            },
            _ => return None,
        };
        Some(primitive.client.placement(primitive))
    }
}

/// [`DispatchTensor::place_in_group`] on a tracked tensor, recorded so its gradient flows back.
#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct PlaceInGroup;

#[cfg(feature = "autodiff")]
impl PlaceInGroup {
    fn record<C: CheckpointStrategy>(
        tensor: FloatTensor<Autodiff<Group>>,
        placement: GroupPlacement,
    ) -> FloatTensor<Autodiff<Group>> {
        let node = tensor.node();
        let inner = tensor.into_primitive();
        let placed = inner.client.clone().place(inner, placement);
        <PlaceInGroup as Backward<Group, 1>>::prepare::<C>(PlaceInGroup, [node])
            .compute_bound()
            .stateless(placed)
    }
}

#[cfg(feature = "autodiff")]
impl<B: Backend> Backward<B, 1> for PlaceInGroup {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

impl BackendTensor<Group> {
    fn place_in_group(self, placement: GroupPlacement) -> Self {
        match self {
            BackendTensor::Float(tensor) => {
                BackendTensor::Float(tensor.client.clone().place(tensor, placement))
            }
            BackendTensor::Int(tensor) => {
                BackendTensor::Int(tensor.client.clone().place(tensor, placement))
            }
            BackendTensor::Bool(tensor) => {
                BackendTensor::Bool(tensor.client.clone().place(tensor, placement))
            }
            BackendTensor::Quantized(_) => {
                panic!("A quantized tensor is not split over a device group")
            }
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => {
                unreachable!("An autodiff tensor is wrapped in its own kind")
            }
        }
    }
}
