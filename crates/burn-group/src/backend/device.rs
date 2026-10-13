use std::sync::atomic::{AtomicU32, Ordering};

use burn_backend::{DeviceId, DeviceOps, DeviceSettings, RouterDeviceType};
use burn_ir::BackendIr;
use burn_router::{get_client, register_scoped_client};
use burn_std::device::Device;

use super::{GroupClient, channel::GroupChannel, interpreter::GroupInterpreter};

/// A group of devices, its members, that a tensor's shards are spread over.
///
/// Like any device, a group lasts as long as the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupDevice {
    index: u16,
}

static NEXT_INDEX: AtomicU32 = AtomicU32::new(0);

impl GroupDevice {
    /// A new group whose members run `B` on `devices`, apart from every other group, even one of
    /// the same devices.
    ///
    /// # Panics
    ///
    /// When `devices` is empty.
    pub fn new<B: BackendIr>(devices: &[B::Device]) -> Self {
        assert!(
            !devices.is_empty(),
            "A device group needs at least one device"
        );
        let index = NEXT_INDEX.fetch_add(1, Ordering::Relaxed);
        let device = Self {
            index: u16::try_from(index).expect("Fewer than 65536 device groups in one process"),
        };
        let interpreter = GroupInterpreter::<B>::new(devices.to_vec());
        let client = GroupClient::new(device, Box::new(interpreter), devices[0].defaults());
        let registration = register_scoped_client::<GroupChannel>(&device, client)
            .expect("A new group's device has no client yet");
        // The client stays registered for the process, as every other device's does.
        core::mem::forget(registration);
        device
    }
}

/// A group exists only once made from its members, so there is no group to default to.
impl Default for GroupDevice {
    fn default() -> Self {
        panic!("A device group is made from its devices, with Device::group or GroupDevice::new")
    }
}

impl Device for GroupDevice {
    fn from_id(device_id: DeviceId) -> Self {
        assert_eq!(
            device_id.type_id,
            u16::from(RouterDeviceType::Group),
            "Not a device group id"
        );
        Self {
            index: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: RouterDeviceType::Group.into(),
            index_id: self.index,
        }
    }
}

impl DeviceOps for GroupDevice {
    fn defaults(&self) -> DeviceSettings {
        get_client::<GroupChannel>(self).settings()
    }
}
