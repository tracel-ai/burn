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
use burn_group::Sharding;

use crate::{BackendTensor, DispatchTensor, DispatchTensorKind, backends::Group};
#[cfg(feature = "autodiff")]
use crate::{DispatchAutodiffContext, GradientCheckpointingStrategy, backends::Autodiff};

impl DispatchTensor {
    /// The same value with another sharding over its device group.
    ///
    /// A tracked autodiff tensor records the move: its gradient flows back through it unchanged,
    /// since a gradient may have any sharding.
    ///
    /// # Panics
    ///
    /// When the tensor is quantized or not on a device group, or no resharding produces
    /// `sharding`: a partial sum out of a tensor that is not one, or a split along a dim the
    /// tensor lacks or one shorter than the group.
    pub fn with_sharding(self, sharding: Sharding) -> Self {
        let DispatchTensor { kind, autodiff } = self;
        let kind = match kind {
            DispatchTensorKind::Group(tensor) => {
                DispatchTensorKind::Group(reshard(tensor, sharding))
            }
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner) => {
                let DispatchAutodiffContext::Enabled(checkpointing) = autodiff else {
                    unreachable!("An autodiff tensor has an enabled autodiff context")
                };
                let DispatchTensorKind::Group(BackendTensor::Autodiff(tensor)) = *inner else {
                    panic!("Only a tensor on a device group has a sharding")
                };
                let resharded = match checkpointing {
                    GradientCheckpointingStrategy::Balanced => {
                        WithSharding::record::<BalancedCheckpointing>(tensor, sharding)
                    }
                    GradientCheckpointingStrategy::Disabled => {
                        WithSharding::record::<NoCheckpointing>(tensor, sharding)
                    }
                };
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Group(
                    BackendTensor::Autodiff(resharded),
                )))
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Only a tensor on a device group has a sharding"),
        };
        DispatchTensor { kind, autodiff }
    }

    /// Where the tensor's shards sit, or `None` when it is not on a device group.
    pub fn sharding(&self) -> Option<Sharding> {
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
        Some(primitive.client.sharding(primitive))
    }
}

/// [`DispatchTensor::with_sharding`] on a tracked tensor, recorded so its gradient flows back.
#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct WithSharding;

#[cfg(feature = "autodiff")]
impl WithSharding {
    fn record<C: CheckpointStrategy>(
        tensor: FloatTensor<Autodiff<Group>>,
        sharding: Sharding,
    ) -> FloatTensor<Autodiff<Group>> {
        let node = tensor.node();
        let inner = tensor.into_primitive();
        let resharded = inner.client.clone().reshard(inner, sharding);
        <WithSharding as Backward<Group, 1>>::prepare::<C>(WithSharding, [node])
            .compute_bound()
            .stateless(resharded)
    }
}

#[cfg(feature = "autodiff")]
impl<B: Backend> Backward<B, 1> for WithSharding {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

fn reshard(tensor: BackendTensor<Group>, sharding: Sharding) -> BackendTensor<Group> {
    match tensor {
        BackendTensor::Float(tensor) => {
            BackendTensor::Float(tensor.client.clone().reshard(tensor, sharding))
        }
        BackendTensor::Int(tensor) => {
            BackendTensor::Int(tensor.client.clone().reshard(tensor, sharding))
        }
        BackendTensor::Bool(tensor) => {
            BackendTensor::Bool(tensor.client.clone().reshard(tensor, sharding))
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
