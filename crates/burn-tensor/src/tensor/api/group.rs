use burn_dispatch::{Dispatch, DispatchDevice, backends::Placement};

use crate::{
    Device, Tensor,
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
    /// When the tensor is not on a [device group](Device::group), or no collective produces
    /// `placement`: a partial sum out of a whole tensor, or a split along a dim shorter than the
    /// group.
    pub fn place(self, placement: Placement) -> Self {
        let (kind, tensor) = self.primitive.into_parts();
        let placed = Dispatch::place(tensor, placement);
        Self::new(match kind {
            BridgeKind::Float => BridgeTensor::float(placed),
            BridgeKind::Int => BridgeTensor::int(placed),
            BridgeKind::Bool => BridgeTensor::bool(placed),
            BridgeKind::QFloat => panic!("A quantized tensor is not split over a device group"),
        })
    }

    /// Where the tensor's shards sit on its device group, or `None` when it is not on one.
    pub fn placement(&self) -> Option<Placement> {
        let (_, tensor) = self.primitive.as_parts();
        Dispatch::placement(tensor)
    }
}

impl Device {
    /// A group of `devices` that each tensor is split over, one device per rank. A tensor made
    /// on it is replicated on every rank until it is [placed](Tensor::place).
    ///
    /// # Panics
    ///
    /// When `devices` is empty or mixes backends, or when its backend cannot run a rank: only
    /// Cube and Flex devices can.
    pub fn group(devices: &[Device]) -> Self {
        let members = devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect();
        Self::new(DispatchDevice::group(members))
    }
}
