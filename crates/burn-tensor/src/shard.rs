//! Sharded compute devices: several devices that a tensor can be split over, for tensor parallelism.
//!
//! Experimental: the API may change, and so far only the forward pass of a Megatron-style MLP
//! keeps its tensors split.
//!
//! ```rust,ignore
//! let device = Device::shard_options(&[Device::cuda(0), Device::cuda(1)]).init()?;
//! ```

use alloc::vec::Vec;

pub use burn_dispatch::backends::ShardError;
use burn_dispatch::{DispatchDevice, backends::ShardCustomOps};

use crate::Device;

/// Which devices form a sharded compute device. Built by [`Device::shard_options`]; nothing is made until
/// [`init`](Self::init).
#[must_use = "shard options do nothing until initialized"]
#[derive(Clone, Debug)]
pub struct ShardOptions {
    devices: Vec<Device>,
    custom_ops: ShardCustomOps,
}

impl ShardOptions {
    /// Every member runs the custom operations of `registry`, a `CustomOpRegistry` of the backend
    /// the members run, as `#[backend_extension]`'s generated `register` fills one.
    ///
    /// A custom operation has no sharding rule, so it gathers its inputs and runs whole on every
    /// member. Calling this again replaces the registry.
    pub fn with_custom_ops(mut self, registry: impl Into<ShardCustomOps>) -> Self {
        self.custom_ops = registry.into();
        self
    }

    /// A new sharded compute device of the devices, its members, that a tensor can be split over, apart from every
    /// other one, even one of the same devices. A tensor made on it is replicated on every
    /// member until it is [sharded](crate::Tensor::with_sharding).
    ///
    /// Make a sharded compute device once and reuse it, as any device: each call registers a new sharded compute device for the
    /// rest of the process.
    ///
    /// Each of its default dtypes is the first of the members' [configured](Device::settings)
    /// ones, in member order, that every member supports.
    ///
    /// Members on different runtimes of one backend form a sharded compute device, and every collective between
    /// them is emulated with transfers staged through the host, so such a sharded compute device is correct and far
    /// slower than one whose members share a runtime.
    ///
    /// # Errors
    ///
    /// When there are no devices, when they mix backends, when they are not Cube or Flex devices,
    /// when no member's default for a dtype runs on every member, when the members quantize
    /// differently, or when the custom operations are another backend's. For autodiff, shard the
    /// plain devices and call `autodiff` on the sharded compute device. Remote devices are left out: tensor parallelism
    /// needs links as fast as one machine's GPU interconnect.
    ///
    /// # Panics
    ///
    /// When the process already has 65,536 sharded compute devices.
    pub fn init(self) -> Result<Device, ShardError> {
        let members = self
            .devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect();
        Ok(Device::new(DispatchDevice::shard(
            members,
            self.custom_ops,
        )?))
    }
}

impl Device {
    /// Options for a sharded compute device of `devices` that a tensor can be split over.
    pub fn shard_options(devices: &[Device]) -> ShardOptions {
        ShardOptions {
            devices: devices.to_vec(),
            custom_ops: ShardCustomOps::default(),
        }
    }
}
