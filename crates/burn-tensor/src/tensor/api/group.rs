use burn_dispatch::{DispatchDevice, backends::GroupPlacement};

use crate::{
    Device, Tensor, check,
    check::TensorCheck,
    kind::Basic,
    ops::{BridgeKind, BridgeTensor},
};

impl<const D: usize, K: Basic> Tensor<D, K> {
    /// The same value, with its shards at `placement` over its device group.
    ///
    /// A tracked tensor records the move, and its gradient flows back through it unchanged. To
    /// place a parameter, place it before requiring its gradient, as with
    /// [`to_device`](Tensor::to_device).
    ///
    /// # Panics
    ///
    /// When the tensor is quantized or not on a [device group](Device::group), when `placement`
    /// is a partial sum and the tensor is not one, or when it splits a dim the tensor lacks or one
    /// shorter than the group.
    pub fn place_in_group(self, placement: GroupPlacement) -> Self {
        check!(TensorCheck::quantized_unsupported(
            "Place In Group",
            self.dtype()
        ));
        check!(TensorCheck::place_in_group::<D>(
            self.group_placement(),
            placement
        ));
        let (kind, tensor) = self.primitive.into_parts();
        let placed = tensor.place_in_group(placement);
        Self::new(match kind {
            BridgeKind::Float => BridgeTensor::float(placed),
            BridgeKind::Int => BridgeTensor::int(placed),
            BridgeKind::Bool => BridgeTensor::bool(placed),
            BridgeKind::QFloat => unreachable!("A quantized tensor is refused above"),
        })
    }

    /// Where the tensor's shards sit on its device group, or `None` when it is not on one.
    pub fn group_placement(&self) -> Option<GroupPlacement> {
        let (_, tensor) = self.primitive.as_parts();
        tensor.group_placement()
    }
}

impl Device {
    /// A group of `devices`, its members, that each tensor is split over. A tensor made on it is
    /// replicated on every member until it is [placed](Tensor::place_in_group).
    ///
    /// # Panics
    ///
    /// When `devices` is empty or mixes backends or runtimes, or when they are not Cube or Flex
    /// devices. For autodiff, group the plain devices and call `autodiff` on the group. Remote
    /// devices are left out: tensor parallelism needs links as fast as one machine's GPU
    /// interconnect.
    pub fn group(devices: &[Device]) -> Self {
        let members = devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect();
        Self::new(DispatchDevice::group(members))
    }
}
