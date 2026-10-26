use burn_dispatch::backends::{Sharding, ShardingError};

use crate::{Tensor, kind::Basic, ops::BridgeTensor};

impl<const D: usize, K: Basic> Tensor<D, K> {
    /// The same value, split over its device group as `sharding` says.
    ///
    /// A tracked tensor records the change, and its gradient flows back through it unchanged. To
    /// shard a parameter, shard it before requiring its gradient, as with
    /// [`to_device`](Tensor::to_device).
    ///
    /// A tensor made from data and split before anything uses it is cut on the host, so no
    /// member ever holds it whole.
    ///
    /// # Panics
    ///
    /// When [`try_with_sharding`](Self::try_with_sharding) refuses the sharding.
    pub fn with_sharding(self, sharding: Sharding) -> Self {
        // Owning the tensor lets the group free the old shards as it reshards, and work in place.
        let (kind, tensor) = self.primitive.into_parts();
        let resharded = tensor
            .try_with_sharding(sharding)
            .unwrap_or_else(|error| panic!("{error}"));
        Self::new(BridgeTensor::from_parts(kind, resharded))
    }

    /// The same value, split over its device group as `sharding` says, or why it cannot be. A
    /// refusal leaves this tensor as it was.
    ///
    /// # Errors
    ///
    /// When the tensor is quantized or not on a [device group](crate::group), when `sharding` is
    /// a partial sum and the tensor is not one, or when it splits a dim the tensor lacks or one
    /// shorter than the group.
    pub fn try_with_sharding(&self, sharding: Sharding) -> Result<Self, ShardingError> {
        let (kind, tensor) = self.primitive.clone().into_parts();
        let resharded = tensor.try_with_sharding(sharding)?;
        Ok(Self::new(BridgeTensor::from_parts(kind, resharded)))
    }

    /// How the tensor is split over its device group, or `None` when it is not on one.
    pub fn sharding(&self) -> Option<Sharding> {
        let (_, tensor) = self.primitive.as_parts();
        tensor.sharding()
    }
}
