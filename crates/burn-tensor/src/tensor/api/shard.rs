use alloc::vec::Vec;

#[cfg(feature = "shard")]
use crate::ops::BridgeTensor;
use crate::{
    Device, Sharding, Shards, ShardsError, Tensor,
    kind::{Autodiff, Basic},
};

impl<const D: usize, K: Autodiff> Tensor<D, K> {
    /// The tensor `shards` make up over the sharded compute device `device`.
    ///
    /// Shard `i` is member `i`'s part, already on that member's device (see
    /// [`Device::shard_members`]): the whole value when replicated or a partial sum, its chunk
    /// of the split dim when split, and its group's chunk when a replicated split. Chunks are
    /// balanced in member order, the first members (or, for a replicated split, every member of
    /// the first groups) holding one more index when the dim does not divide. Nothing moves between members, so a caller
    /// that makes each shard on its member never holds the whole tensor anywhere.
    ///
    /// Quantized shards of a split must share a per-tensor or two-level global scale, which the
    /// whole keeps once: shards quantized each on their own are checked only when gathered, and
    /// a read or an op that gathers them fails.
    ///
    /// A shard's autodiff tracking is dropped. On an autodiff sharded compute device the tensor is
    /// a new leaf: require its gradient afterwards.
    ///
    /// # Errors
    ///
    /// When `device` is not a sharded compute device, which needs the `shard` feature, when there
    /// is not one shard per member, when shard `i` is not on member `i`'s device, when the shards
    /// differ in dtype or quantization scheme, when a partial sum is not of floats or ints, when a
    /// replicated split's groups do not divide the members or are one or every member, or when a
    /// shard's shape is not its
    /// member's part, a quantized chunk holding whole quantization blocks and packed words.
    pub fn from_shards(shards: Shards<Self>, device: &Device) -> Result<Self, ShardsError> {
        #[cfg(feature = "shard")]
        {
            let mut kind = None;
            let shards = shards.map(|shard| {
                let (shard_kind, tensor) = shard.without_autodiff().primitive.into_parts();
                kind.get_or_insert(shard_kind);
                tensor
            });
            let plain = device.clone().inner();
            let joined = burn_dispatch::DispatchTensor::from_shards(shards, plain.as_dispatch())?;
            let kind = kind.expect("A sharded compute device takes a shard per member");
            let tensor = Self::new(BridgeTensor::from_parts(kind, joined));
            #[cfg(feature = "autodiff")]
            if let Some(strategy) = device.gradient_checkpointing_strategy() {
                return Ok(tensor
                    .autodiff()
                    .with_gradient_checkpointing_strategy(strategy));
            }
            Ok(tensor)
        }
        #[cfg(not(feature = "shard"))]
        {
            let _ = (shards, device);
            Err(ShardsError::NotSharded)
        }
    }

    /// Each member's shard of the tensor, on that member's device, and how they make up its
    /// value. The shards are not tracked by autodiff.
    ///
    /// # Panics
    ///
    /// When the tensor is not on a sharded compute device.
    pub fn into_shards(self) -> Shards<Self> {
        #[cfg(feature = "shard")]
        {
            let (kind, tensor) = self.without_autodiff().primitive.into_parts();
            tensor
                .into_shards()
                .map(|shard| Self::new(BridgeTensor::from_parts(kind, shard)))
        }
        #[cfg(not(feature = "shard"))]
        {
            let _ = self;
            panic!(
                "{} (the `shard` feature is not enabled)",
                ShardsError::NotSharded
            )
        }
    }
}

impl<const D: usize, K: Basic> Tensor<D, K> {
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
    /// The members of this sharded compute device, in member order, or `None` when it is not one:
    /// the devices the shards of [`Tensor::from_shards`] are made on. They are plain devices, an
    /// autodiff sharded compute device's included.
    pub fn shard_members(&self) -> Option<Vec<Device>> {
        #[cfg(feature = "shard")]
        {
            self.as_dispatch()
                .shard_members()
                .map(|members| members.into_iter().map(Device::new).collect())
        }
        #[cfg(not(feature = "shard"))]
        None
    }
}

#[cfg(all(test, not(feature = "shard")))]
mod tests {
    use alloc::vec;

    use crate::{Device, Sharding, Shards, ShardsError, Tensor};

    #[test]
    fn without_the_shard_feature_nothing_is_sharded() {
        let device = Device::default();
        let tensor = Tensor::<1>::zeros([4], &device);

        assert_eq!(tensor.sharding(), None);
        assert_eq!(device.shard_members(), None);
        assert_eq!(
            Tensor::<1>::from_shards(
                Shards {
                    shards: vec![tensor],
                    sharding: Sharding::Replicated,
                },
                &device
            )
            .err(),
            Some(ShardsError::NotSharded)
        );
    }
}
