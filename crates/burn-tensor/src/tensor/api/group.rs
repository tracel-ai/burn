use burn_dispatch::backends::{Sharding, ShardingError};

use crate::{
    Tensor,
    kind::Basic,
    ops::{BridgeKind, BridgeTensor},
};

impl<const D: usize, K: Basic> Tensor<D, K> {
    /// The same value, split over its device group as `sharding` says.
    ///
    /// A tracked tensor records the change, and its gradient flows back through it unchanged. To
    /// shard a parameter, shard it before requiring its gradient, as with
    /// [`to_device`](Tensor::to_device).
    ///
    /// # Panics
    ///
    /// When [`try_with_sharding`](Self::try_with_sharding) refuses the sharding.
    pub fn with_sharding(self, sharding: Sharding) -> Self {
        self.try_with_sharding(sharding)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// The same value, split over its device group as `sharding` says, or why it cannot be.
    ///
    /// # Errors
    ///
    /// When the tensor is quantized or not on a [device group](crate::group), when `sharding` is
    /// a partial sum and the tensor is not one, or when it splits a dim the tensor lacks or one
    /// shorter than the group.
    pub fn try_with_sharding(self, sharding: Sharding) -> Result<Self, ShardingError> {
        let (kind, tensor) = self.primitive.into_parts();
        let resharded = tensor.try_with_sharding(sharding)?;
        Ok(Self::new(match kind {
            BridgeKind::Float => BridgeTensor::float(resharded),
            BridgeKind::Int => BridgeTensor::int(resharded),
            BridgeKind::Bool => BridgeTensor::bool(resharded),
            BridgeKind::QFloat => unreachable!("A quantized tensor is refused as it is resharded"),
        }))
    }

    /// How the tensor is split over its device group, or `None` when it is not on one.
    pub fn sharding(&self) -> Option<Sharding> {
        let (_, tensor) = self.primitive.as_parts();
        tensor.sharding()
    }
}
