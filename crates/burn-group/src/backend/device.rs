use std::any::TypeId;

use burn_backend::{DeviceId, DeviceOps, DeviceSettings, RouterDeviceType};
use burn_ir::BackendIr;
use burn_std::{device::Device, sync::Mutex};

use super::interpreter::{DynGroupInterpreter, GroupInterpreter};

/// A group of devices, its members, that a tensor's shards are spread over.
///
/// The members live in a process-wide registry behind the index, so the device fits in a
/// [`DeviceId`] like any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupDevice {
    index: u16,
}

impl GroupDevice {
    /// A group whose members run `B` on `devices`. The same devices and backend are the same
    /// group.
    ///
    /// # Panics
    ///
    /// When `devices` is empty.
    pub fn new<B: BackendIr>(devices: &[B::Device]) -> Self {
        assert!(
            !devices.is_empty(),
            "A device group needs at least one device"
        );
        let group = RegisteredGroup {
            devices: devices.iter().map(DeviceOps::id).collect(),
            backend: TypeId::of::<B>(),
            settings: devices[0].defaults(),
            interpreter: GroupInterpreter::<B>::boxed,
        };
        let mut groups = GROUPS.lock();
        let index = match groups
            .iter()
            .position(|registered| registered.is_like(&group))
        {
            Some(index) => index,
            None => {
                groups.push(group);
                groups.len() - 1
            }
        };
        Self {
            index: u16::try_from(index).expect("Fewer than 65536 device groups"),
        }
    }

    /// The number of devices in the group.
    pub fn num_members(&self) -> usize {
        RegisteredGroup::read(self, |group| group.devices.len())
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
        RegisteredGroup::read(self, |group| group.settings)
    }
}

static GROUPS: Mutex<Vec<RegisteredGroup>> = Mutex::new(Vec::new());

/// What a [`GroupDevice`] stands for: its devices, their backend and how to run them.
#[derive(Clone, Debug)]
pub struct RegisteredGroup {
    devices: Vec<DeviceId>,
    backend: TypeId,
    settings: DeviceSettings,
    interpreter: fn(&[DeviceId]) -> Box<dyn DynGroupInterpreter>,
}

impl RegisteredGroup {
    pub fn new(device: &GroupDevice) -> Self {
        Self::read(device, Clone::clone)
    }

    /// A new interpreter for the group's members, holding no tensors yet.
    pub fn interpreter(&self) -> Box<dyn DynGroupInterpreter> {
        (self.interpreter)(&self.devices)
    }

    fn read<R>(device: &GroupDevice, read: impl FnOnce(&Self) -> R) -> R {
        let groups = GROUPS.lock();
        let group = groups
            .get(usize::from(device.index))
            .expect("A device group is made with GroupDevice::new before it is used");
        read(group)
    }

    fn is_like(&self, other: &Self) -> bool {
        self.devices == other.devices && self.backend == other.backend
    }
}
