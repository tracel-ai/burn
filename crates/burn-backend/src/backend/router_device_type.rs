/// The type id a device behind burn-router puts in its [`DeviceId`](super::DeviceId).
///
/// It names no cubecl runtime, so the device never shares a cubecl runner thread with a local
/// GPU, and it fits the low byte of a backend's type id that burn-dispatch keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RouterDeviceType {
    /// A device on a remote server.
    Remote = 254,
    /// A device that records operations instead of running them.
    Capture = 255,
}

impl RouterDeviceType {
    /// The type id to put in a [`DeviceId`](super::DeviceId).
    pub const fn type_id(self) -> u16 {
        self as u16
    }
}

#[cfg(all(test, feature = "cubecl"))]
mod tests {
    use super::*;
    use crate::{DeviceId, cubecl::RuntimeId};

    #[test]
    fn a_router_device_type_names_no_cubecl_runtime() {
        for device_type in [RouterDeviceType::Remote, RouterDeviceType::Capture] {
            let id = DeviceId::new(device_type.type_id(), 0);

            assert_eq!(RuntimeId::of_device_id(id).ok(), None, "{device_type:?}");
        }
    }
}
