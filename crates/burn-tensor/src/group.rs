//! Groups of devices that a tensor can be split over, for tensor parallelism.
//!
//! Experimental: the API may change, and so far only the forward pass of a Megatron-style MLP
//! keeps its tensors split.
//!
//! ```rust,ignore
//! let device = Device::group_options(&[Device::cuda(0), Device::cuda(1)]).init()?;
//! ```

use alloc::vec::Vec;

use burn_dispatch::DispatchDevice;
pub use burn_dispatch::backends::GroupError;

use crate::Device;

/// Which devices form a group. Built by [`Device::group_options`]; nothing is made until
/// [`init`](Self::init).
#[must_use = "group options do nothing until initialized"]
#[derive(Clone, Debug)]
pub struct GroupOptions {
    devices: Vec<Device>,
}

impl GroupOptions {
    /// A new group of the devices, its members, that a tensor can be split over, apart from every
    /// other group, even one of the same devices. A tensor made on it is replicated on every
    /// member until it is [sharded](crate::Tensor::with_sharding).
    ///
    /// Make a group once and reuse it, as any device: each call registers a new group for the
    /// rest of the process.
    ///
    /// Each of its default dtypes is the first of the members' [configured](Device::settings)
    /// ones, in member order, that every member supports.
    ///
    /// Members on different runtimes of one backend form a group, and every collective between
    /// them is emulated with transfers staged through the host, so such a group is correct and far
    /// slower than one whose members share a runtime.
    ///
    /// # Errors
    ///
    /// When there are no devices, when they mix backends, when they are not Cube or Flex devices,
    /// when no member's default for a dtype runs on every member, or when the members quantize
    /// differently. For autodiff, group the plain
    /// devices and call `autodiff` on the group. Remote devices are left out: tensor parallelism
    /// needs links as fast as one machine's GPU interconnect.
    ///
    /// # Panics
    ///
    /// When the process already has 65,536 groups.
    pub fn init(self) -> Result<Device, GroupError> {
        let members = self
            .devices
            .iter()
            .map(|device| device.as_dispatch().clone())
            .collect();
        Ok(Device::new(DispatchDevice::group(members)?))
    }
}

impl Device {
    /// Options for a group of `devices` that a tensor can be split over.
    pub fn group_options(devices: &[Device]) -> GroupOptions {
        GroupOptions {
            devices: devices.to_vec(),
        }
    }
}
