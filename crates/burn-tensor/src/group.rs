//! Splitting each tensor over a group of devices, for tensor parallelism.
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
    /// A new group of the devices, its members, that each tensor is split over, apart from every
    /// other group, even one of the same devices. A tensor made on it is replicated on every
    /// member until it is [sharded](crate::Tensor::with_sharding).
    ///
    /// Make a group once and reuse it, as any device: each call registers a new group for the
    /// rest of the process.
    ///
    /// Each of its default dtypes is the first member's [configured](Device::settings) one that
    /// every member supports.
    ///
    /// # Errors
    ///
    /// When there are no devices, when they mix backends or runtimes, when they are not Cube or
    /// Flex devices, or when no member's default for a dtype runs on every member. For autodiff,
    /// group the plain devices and call `autodiff` on the group. Remote devices are left out:
    /// tensor parallelism needs links as fast as one machine's GPU interconnect.
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
    /// Options for a group of `devices` that splits each tensor over them.
    pub fn group_options(devices: &[Device]) -> GroupOptions {
        GroupOptions {
            devices: devices.to_vec(),
        }
    }
}
