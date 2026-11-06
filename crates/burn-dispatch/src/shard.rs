#[cfg(any(cube_backend, feature = "flex"))]
use alloc::vec::Vec;
#[cfg(any(cube_backend, feature = "flex"))]
use burn_backend::{DType, TensorMetadata};
#[cfg(any(cube_backend, feature = "flex"))]
use burn_ir::{BackendIr, HandleKind};
use burn_shard::{Sharding, Shards, ShardsError};

use crate::{
    BackendTensor, DispatchDevice, DispatchKindConversion, DispatchTensor, DispatchTensorKind,
    backends::ShardedCompute,
};

impl DispatchTensor {
    /// The tensor `shards` make up over the sharded compute device `device`: one shard per
    /// member, each already on its member. Nothing moves between members.
    ///
    /// # Errors
    ///
    /// When `device` is not a plain sharded compute device, or as
    /// [`ShardedComputeDevice::join_shards`](burn_shard::ShardedComputeDevice::join_shards).
    pub fn from_shards(
        shards: Shards<DispatchTensor>,
        device: &DispatchDevice,
    ) -> Result<Self, ShardsError> {
        let device = match device {
            DispatchDevice::ShardedCompute(device) => device,
            #[allow(unreachable_patterns)]
            _ => return Err(ShardsError::NotSharded),
        };
        let tensor = match shards.shards.first().map(|shard| &shard.kind) {
            #[cfg(cube_backend)]
            Some(DispatchTensorKind::Cube(_)) => {
                BackendTensor::from_member_shards::<crate::backends::Cube>(shards, device)
            }
            #[cfg(feature = "flex")]
            Some(DispatchTensorKind::Flex(_)) => {
                BackendTensor::from_member_shards::<crate::backends::Flex>(shards, device)
            }
            Some(_) => Err(ShardsError::Device { shard: 0 }),
            None => Err(ShardsError::Count {
                shards: 0,
                num_members: device.num_members(),
            }),
        }?;
        Ok(<Self as DispatchKindConversion<ShardedCompute>>::from_backend(tensor))
    }

    /// Each member's shard of the tensor, in member order, and how they make up its value.
    ///
    /// # Panics
    ///
    /// When the tensor is not on a plain sharded compute device.
    #[cfg_attr(not(any(cube_backend, feature = "flex")), allow(unused_variables))]
    pub fn into_shards(self) -> Shards<DispatchTensor> {
        let tensor = match self.kind {
            DispatchTensorKind::ShardedCompute(tensor) => tensor,
            #[allow(unreachable_patterns)]
            _ => panic!("{}", ShardsError::NotSharded),
        };
        #[cfg(cube_backend)]
        if tensor.device().members::<crate::backends::Cube>().is_some() {
            return tensor.into_member_shards::<crate::backends::Cube>();
        }
        #[cfg(feature = "flex")]
        if tensor.device().members::<crate::backends::Flex>().is_some() {
            return tensor.into_member_shards::<crate::backends::Flex>();
        }
        unreachable!("A sharded compute device's members are Cube or Flex devices")
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

#[cfg(any(cube_backend, feature = "flex"))]
impl BackendTensor<ShardedCompute> {
    /// The tensor of the sharded compute device `shards` of the members' backend `B` make up.
    fn from_member_shards<B: BackendIr>(
        Shards { shards, sharding }: Shards<DispatchTensor>,
        device: &burn_shard::ShardedComputeDevice,
    ) -> Result<Self, ShardsError>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let handles = shards
            .into_iter()
            .enumerate()
            .map(|(shard, tensor)| {
                <DispatchTensor as DispatchKindConversion<B>>::try_into_backend(tensor)
                    .ok()
                    .and_then(BackendTensor::into_member_handle)
                    .ok_or(ShardsError::Device { shard })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = device.join_shards::<B>(Shards {
            shards: handles,
            sharding,
        })?;
        Ok(match tensor.dtype() {
            DType::QFloat(_) => Self::Quantized(tensor),
            DType::Bool(_) => Self::Bool(tensor),
            dtype if dtype.is_float() => Self::Float(tensor),
            _ => Self::Int(tensor),
        })
    }

    /// Each member's shard of the tensor, whose members run `B`.
    fn into_member_shards<B: BackendIr>(self) -> Shards<DispatchTensor>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let tensor = match self {
            Self::Float(tensor)
            | Self::Int(tensor)
            | Self::Bool(tensor)
            | Self::Quantized(tensor) => tensor,
            #[cfg(feature = "autodiff")]
            Self::Autodiff(_) => unreachable!("An autodiff tensor is wrapped in its own kind"),
        };
        tensor
            .client
            .clone()
            .take_shards::<B>(tensor)
            .map(|handle| {
                <DispatchTensor as DispatchKindConversion<B>>::from_backend(BackendTensor::from(
                    handle,
                ))
            })
    }
}

#[cfg(any(cube_backend, feature = "flex"))]
impl<B: BackendIr> BackendTensor<B> {
    /// The tensor as its member's handle, or `None` for a tracked one, which no member holds.
    fn into_member_handle(self) -> Option<HandleKind<B>> {
        Some(match self {
            Self::Float(tensor) => HandleKind::Float(tensor),
            Self::Int(tensor) => HandleKind::Int(tensor),
            Self::Bool(tensor) => HandleKind::Bool(tensor),
            Self::Quantized(tensor) => HandleKind::Quantized(tensor),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(_) => return None,
        })
    }
}

#[cfg(any(cube_backend, feature = "flex"))]
impl<B: BackendIr> From<HandleKind<B>> for BackendTensor<B> {
    fn from(handle: HandleKind<B>) -> Self {
        match handle {
            HandleKind::Float(tensor) => Self::Float(tensor),
            HandleKind::Int(tensor) => Self::Int(tensor),
            HandleKind::Bool(tensor) => Self::Bool(tensor),
            HandleKind::Quantized(tensor) => Self::Quantized(tensor),
        }
    }
}
