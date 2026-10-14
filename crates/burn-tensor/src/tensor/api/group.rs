use burn_dispatch::{DispatchDevice, backends::Sharding};

use crate::{
    Device, Tensor, check,
    check::TensorCheck,
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
    /// When the tensor is quantized or not on a [device group](Device::group), when `sharding`
    /// is a partial sum and the tensor is not one, or when it splits a dim the tensor lacks or one
    /// shorter than the group.
    pub fn with_sharding(self, sharding: Sharding) -> Self {
        check!(TensorCheck::quantized_unsupported(
            "With Sharding",
            self.dtype()
        ));
        check!(TensorCheck::with_sharding::<D>(self.sharding(), sharding));
        let (kind, tensor) = self.primitive.into_parts();
        let resharded = tensor.with_sharding(sharding);
        Self::new(match kind {
            BridgeKind::Float => BridgeTensor::float(resharded),
            BridgeKind::Int => BridgeTensor::int(resharded),
            BridgeKind::Bool => BridgeTensor::bool(resharded),
            BridgeKind::QFloat => unreachable!("A quantized tensor is refused above"),
        })
    }

    /// How the tensor is split over its device group, or `None` when it is not on one.
    pub fn sharding(&self) -> Option<Sharding> {
        let (_, tensor) = self.primitive.as_parts();
        tensor.sharding()
    }
}

impl Device {
    /// A new group of `devices`, its members, that each tensor is split over, apart from every
    /// other group, even one of the same devices. A tensor made on it is replicated on every
    /// member until it is [sharded](Tensor::with_sharding).
    ///
    /// Make a group once and reuse it, as any device: each call registers a new group for the
    /// rest of the process.
    ///
    /// Each of its default dtypes is the first member's [configured](Device::settings) one that
    /// every member supports.
    ///
    /// # Panics
    ///
    /// When `devices` is empty or mixes backends or runtimes, when they are not Cube or Flex
    /// devices, or when no member's default for a dtype runs on every member. For autodiff, group
    /// the plain devices and call `autodiff` on the group. Remote devices are left out: tensor
    /// parallelism needs links as fast as one machine's GPU interconnect.
    pub fn group(devices: &[Device]) -> Self {
        let members = devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect();
        Self::new(DispatchDevice::group(members))
    }
}
