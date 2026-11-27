//! The devices an instrumented run launches on.

use std::fmt;

use burn_tensor::Device;
use cubecl_common::device::DeviceId;

/// The devices a run launches on: one, or several that take turns, each one cubecl device.
#[derive(Clone)]
pub struct DeviceList {
    /// Walked in the caller's order to seed, drain and number the members.
    devices: Vec<Device>,
}

impl DeviceList {
    /// # Panics
    ///
    /// If `devices` is empty, or one of several is not one cubecl device.
    pub fn new(devices: &[Device]) -> Self {
        assert!(!devices.is_empty(), "an instrument watches a device");
        assert!(
            devices.len() == 1
                || devices
                    .iter()
                    .all(|device| device.as_dispatch().member_ids().count() == 1),
            "devices taking turns are timed one member each, so each is one cubecl device"
        );
        Self {
            devices: devices.to_vec(),
        }
    }

    /// The device of a list of one.
    pub fn single(&self) -> Option<&Device> {
        match self.devices.as_slice() {
            [device] => Some(device),
            _ => None,
        }
    }

    /// The device whose settings the run takes its defaults from.
    pub fn first(&self) -> &Device {
        &self.devices[0]
    }

    /// Whether the devices take turns.
    pub fn several(&self) -> bool {
        self.devices.len() > 1
    }

    /// Whether the run is on one sharded compute device of several members, which work at once.
    pub fn sharded(&self) -> bool {
        (self.single())
            .and_then(Device::shard_members)
            .is_some_and(|members| members.len() > 1)
    }

    /// The cubecl devices behind every device, in order: see `DispatchDevice::member_ids`.
    pub fn member_ids(&self) -> impl Iterator<Item = DeviceId> + '_ {
        (self.devices.iter()).flat_map(|device| device.as_dispatch().member_ids())
    }

    /// The fusion workers behind every device, in order: see `DispatchDevice::fusion_workers`.
    pub fn fusion_workers(&self) -> impl Iterator<Item = DeviceId> + '_ {
        (self.devices.iter()).flat_map(|device| device.as_dispatch().fusion_workers())
    }

    pub fn seed(&self, seed: u64) {
        for device in &self.devices {
            device.seed(seed);
        }
    }

    /// # Panics
    ///
    /// With `expectation` if a device stops answering.
    pub fn sync(&self, expectation: &str) {
        for device in &self.devices {
            device.sync().expect(expectation);
        }
    }
}

/// One device as it prints itself, several as their list.
impl fmt::Debug for DeviceList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.single() {
            Some(device) => device.fmt(f),
            None => self.devices.fmt(f),
        }
    }
}
