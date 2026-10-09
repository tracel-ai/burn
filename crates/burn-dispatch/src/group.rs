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

use crate::{BackendTensor, Dispatch, DispatchTensor, DispatchTensorKind, backends::Group};
#[cfg(feature = "autodiff")]
use crate::{DispatchAutodiffContext, GradientCheckpointingStrategy, backends::Autodiff};

impl Dispatch {
    /// The same value at another placement over its device group.
    ///
    /// A tracked autodiff tensor records the move: its gradient flows back through it unchanged,
    /// since a gradient may sit at any placement.
    ///
    /// # Panics
    ///
    /// When the tensor is not on a device group, or no collective produces `placement`: a
    /// partial sum out of a whole tensor, or a split along a dim shorter than the group.
    pub fn place_in_group(tensor: DispatchTensor, placement: GroupPlacement) -> DispatchTensor {
        let DispatchTensor { kind, autodiff } = tensor;
        let kind = match kind {
            DispatchTensorKind::Group(tensor) => {
                DispatchTensorKind::Group(place_backend_tensor(tensor, placement))
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
                        Place::record::<BalancedCheckpointing>(tensor, placement)
                    }
                    GradientCheckpointingStrategy::Disabled => {
                        Place::record::<NoCheckpointing>(tensor, placement)
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
    pub fn group_placement(tensor: &DispatchTensor) -> Option<GroupPlacement> {
        let primitive = match &tensor.kind {
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

/// The autodiff record of a move between placements.
#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct Place;

#[cfg(feature = "autodiff")]
impl Place {
    fn record<C: CheckpointStrategy>(
        tensor: FloatTensor<Autodiff<Group>>,
        placement: GroupPlacement,
    ) -> FloatTensor<Autodiff<Group>> {
        let node = tensor.node();
        let inner = tensor.into_primitive();
        let placed = inner.client.clone().place(inner, placement);
        <Place as Backward<Group, 1>>::prepare::<C>(Place, [node])
            .compute_bound()
            .stateless(placed)
    }
}

#[cfg(feature = "autodiff")]
impl<B: Backend> Backward<B, 1> for Place {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

fn place_backend_tensor(
    tensor: BackendTensor<Group>,
    placement: GroupPlacement,
) -> BackendTensor<Group> {
    match tensor {
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
        BackendTensor::Autodiff(_) => unreachable!("An autodiff tensor is wrapped in its own kind"),
    }
}
