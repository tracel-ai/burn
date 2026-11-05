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
use burn_shard::{Sharding, ShardingError};

use crate::{BackendTensor, DispatchTensor, DispatchTensorKind, backends::ShardedCompute};
#[cfg(feature = "autodiff")]
use crate::{DispatchAutodiffContext, GradientCheckpointingStrategy, backends::Autodiff};

impl DispatchTensor {
    /// The same value with another sharding over its sharded compute device, or why it cannot take it.
    ///
    /// A tracked autodiff tensor records the move: its gradient flows back through it unchanged,
    /// since a gradient may have any sharding.
    pub fn try_with_sharding(self, sharding: Sharding) -> Result<Self, ShardingError> {
        let DispatchTensor { kind, autodiff } = self;
        let kind = match kind {
            DispatchTensorKind::ShardedCompute(tensor) => {
                DispatchTensorKind::ShardedCompute(tensor.reshard(sharding)?)
            }
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner) => {
                let DispatchAutodiffContext::Enabled(checkpointing) = autodiff else {
                    unreachable!("An autodiff tensor has an enabled autodiff context")
                };
                let DispatchTensorKind::ShardedCompute(BackendTensor::Autodiff(tensor)) = *inner
                else {
                    return Err(ShardingError::NotSharded);
                };
                let resharded = match checkpointing {
                    GradientCheckpointingStrategy::Balanced => {
                        WithSharding::record::<BalancedCheckpointing>(tensor, sharding)
                    }
                    GradientCheckpointingStrategy::Disabled => {
                        WithSharding::record::<NoCheckpointing>(tensor, sharding)
                    }
                }?;
                DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::ShardedCompute(
                    BackendTensor::Autodiff(resharded),
                )))
            }
            #[allow(unreachable_patterns)]
            _ => return Err(ShardingError::NotSharded),
        };
        Ok(DispatchTensor { kind, autodiff })
    }

    /// Where the tensor's shards sit, or `None` when it is not on a sharded compute device.
    pub fn sharding(&self) -> Option<Sharding> {
        let primitive = match &self.kind {
            DispatchTensorKind::ShardedCompute(
                BackendTensor::Float(tensor)
                | BackendTensor::Int(tensor)
                | BackendTensor::Bool(tensor)
                | BackendTensor::Quantized(tensor),
            ) => tensor,
            #[cfg(feature = "autodiff")]
            DispatchTensorKind::Autodiff(inner) => match &**inner {
                DispatchTensorKind::ShardedCompute(BackendTensor::Autodiff(tensor)) => {
                    tensor.primitive()
                }
                _ => return None,
            },
            #[allow(unreachable_patterns)]
            _ => return None,
        };
        Some(primitive.client.sharding(primitive))
    }
}

/// [`DispatchTensor::try_with_sharding`] on a tracked tensor, recorded so its gradient flows back.
#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct WithSharding;

#[cfg(feature = "autodiff")]
impl WithSharding {
    fn record<C: CheckpointStrategy>(
        tensor: FloatTensor<Autodiff<ShardedCompute>>,
        sharding: Sharding,
    ) -> Result<FloatTensor<Autodiff<ShardedCompute>>, ShardingError> {
        let node = tensor.node();
        let inner = tensor.into_primitive();
        let resharded = inner.client.clone().reshard(inner, sharding)?;
        Ok(
            <WithSharding as Backward<ShardedCompute, 1>>::prepare::<C>(WithSharding, [node])
                .compute_bound()
                .stateless(resharded),
        )
    }
}

#[cfg(feature = "autodiff")]
impl<B: Backend> Backward<B, 1> for WithSharding {
    type State = ();

    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

impl BackendTensor<ShardedCompute> {
    fn reshard(self, sharding: Sharding) -> Result<Self, ShardingError> {
        Ok(match self {
            BackendTensor::Float(tensor) => {
                BackendTensor::Float(tensor.client.clone().reshard(tensor, sharding)?)
            }
            BackendTensor::Int(tensor) => {
                BackendTensor::Int(tensor.client.clone().reshard(tensor, sharding)?)
            }
            BackendTensor::Bool(tensor) => {
                BackendTensor::Bool(tensor.client.clone().reshard(tensor, sharding)?)
            }
            BackendTensor::Quantized(tensor) => {
                BackendTensor::Quantized(tensor.client.clone().reshard(tensor, sharding)?)
            }
            #[cfg(feature = "autodiff")]
            BackendTensor::Autodiff(_) => {
                unreachable!("An autodiff tensor is wrapped in its own kind")
            }
        })
    }
}
