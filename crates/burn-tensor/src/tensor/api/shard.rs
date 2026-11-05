#[cfg(feature = "shard")]
use crate::ops::BridgeTensor;
use crate::{Device, Shape, Sharding, ShardingError, Tensor, kind::Basic};

impl<const D: usize, K: Basic> Tensor<D, K> {
    /// The same value, split over its sharded compute device as `sharding` says.
    ///
    /// A tracked tensor records the change, and its gradient flows back through it unchanged. To
    /// shard a parameter, shard it before requiring its gradient, as with
    /// [`to_device`](Tensor::to_device).
    ///
    /// Splitting a tensor made from data before any op uses it uploads each member only its
    /// chunk.
    ///
    /// # Panics
    ///
    /// When [`try_with_sharding`](Self::try_with_sharding) refuses the sharding.
    pub fn with_sharding(self, sharding: Sharding) -> Self {
        #[cfg(feature = "shard")]
        {
            // Owning the tensor lets the sharded compute device free the old shards as it
            // reshards, and work in place.
            let (kind, tensor) = self.primitive.into_parts();
            let resharded = tensor
                .try_with_sharding(sharding)
                .unwrap_or_else(|error| panic!("{error}"));
            Self::new(BridgeTensor::from_parts(kind, resharded))
        }
        #[cfg(not(feature = "shard"))]
        {
            let _ = (self, sharding);
            panic!(
                "{} (the `shard` feature is not enabled)",
                ShardingError::NotSharded
            )
        }
    }

    /// The same value, split over its sharded compute device as `sharding` says, or why it cannot
    /// be. A refusal leaves this tensor as it was.
    ///
    /// # Errors
    ///
    /// When the tensor is not on a sharded compute device, which needs the `shard` feature, when
    /// `sharding` is a partial sum and the tensor is not one, when it splits a dim the tensor lacks
    /// or one shorter than the sharded compute device, or, for a quantized tensor, when a member's
    /// chunk would not hold whole quantization blocks, or whole packed words along its packed axis:
    /// its codes and scales are cut as they are.
    pub fn try_with_sharding(&self, sharding: Sharding) -> Result<Self, ShardingError> {
        #[cfg(feature = "shard")]
        {
            let (kind, tensor) = self.primitive.clone().into_parts();
            let resharded = tensor.try_with_sharding(sharding)?;
            Ok(Self::new(BridgeTensor::from_parts(kind, resharded)))
        }
        #[cfg(not(feature = "shard"))]
        {
            let _ = sharding;
            Err(ShardingError::NotSharded)
        }
    }

    /// How the tensor is split over its sharded compute device, or `None` when it is not on one.
    pub fn sharding(&self) -> Option<Sharding> {
        #[cfg(feature = "shard")]
        {
            let (_, tensor) = self.primitive.as_parts();
            tensor.sharding()
        }
        #[cfg(not(feature = "shard"))]
        None
    }
}

impl Device {
    /// Why a tensor of `shape` made on this device cannot take `sharding`, if it cannot, without
    /// making the tensor.
    ///
    /// # Errors
    ///
    /// As [`try_with_sharding`](Tensor::try_with_sharding) for a replicated, unquantized tensor.
    pub fn check_sharding(&self, shape: &Shape, sharding: Sharding) -> Result<(), ShardingError> {
        #[cfg(feature = "shard")]
        {
            self.as_dispatch().check_sharding(shape, sharding)
        }
        #[cfg(not(feature = "shard"))]
        {
            let _ = (shape, sharding);
            Err(ShardingError::NotSharded)
        }
    }
}

#[cfg(all(test, not(feature = "shard")))]
mod tests {
    use crate::{Device, Shape, Sharding, ShardingError, Tensor};

    #[test]
    fn without_the_shard_feature_nothing_is_sharded() {
        let device = Device::default();
        let tensor = Tensor::<1>::zeros([4], &device);

        assert_eq!(tensor.sharding(), None);
        assert_eq!(
            tensor.try_with_sharding(Sharding::Split { dim: 0 }).err(),
            Some(ShardingError::NotSharded)
        );
        assert_eq!(
            device.check_sharding(&Shape::new([4]), Sharding::Replicated),
            Err(ShardingError::NotSharded)
        );
    }
}
