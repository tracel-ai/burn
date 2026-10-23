/// The type id a device behind burn-router puts in its [`DeviceId`](super::DeviceId).
///
/// It names no cubecl runtime, so the device never shares a cubecl runner thread with a local
/// GPU. It is a `u8` because burn-dispatch keeps only the low byte of a backend's type id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RouterDeviceType {
    /// A group of devices that a tensor's shards are spread over.
    Group = 253,
    /// A device on a remote server.
    Remote = 254,
    /// A device that records operations instead of running them.
    Capture = 255,
}

impl From<RouterDeviceType> for u16 {
    fn from(device_type: RouterDeviceType) -> Self {
        u16::from(device_type as u8)
    }
}

#[cfg(all(test, feature = "cubecl"))]
mod tests {
    use super::*;
    use crate::{DeviceId, cubecl::RuntimeId};

    /// Every variant: the match stops compiling when one is added, until it is listed here.
    fn every_router_device_type() -> [RouterDeviceType; 3] {
        match RouterDeviceType::Remote {
            RouterDeviceType::Group | RouterDeviceType::Remote | RouterDeviceType::Capture => {}
        }
        [
            RouterDeviceType::Group,
            RouterDeviceType::Remote,
            RouterDeviceType::Capture,
        ]
    }

    #[test]
    fn a_router_device_type_names_no_cubecl_runtime() {
        for device_type in every_router_device_type() {
            let id = DeviceId::new(device_type.into(), 0);

            assert_eq!(RuntimeId::of_device_id(id).ok(), None, "{device_type:?}");
        }
    }
}
