use std::{
    fmt,
    sync::atomic::{AtomicU32, Ordering},
};

use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, RouterDeviceType};
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

/// Why a set of devices cannot form a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupError {
    /// A group needs at least one member.
    NoDevices,
    /// Every member of a group runs one backend.
    MixedBackends,
    /// Every member of a group runs one runtime of its backend.
    MixedRuntimes,
    /// A group is made from plain devices: group them, then make the group autodiff.
    AutodiffMember,
    /// Tensor parallelism needs links as fast as one machine's GPU interconnect, which a remote
    /// device does not have.
    RemoteMember,
    /// Only Cube and Flex devices form a group.
    UnsupportedBackend {
        /// The device that cannot be a member.
        device: String,
    },
    /// No member's default for a dtype runs on every member.
    NoSharedDefault {
        /// Each member's default for that dtype, in member order.
        candidates: Vec<DType>,
    },
    /// A group takes one settings per member.
    SettingsPerMember {
        /// How many devices were given.
        devices: usize,
        /// How many settings were given.
        settings: usize,
    },
}

impl GroupDevice {
    /// A new group whose members run `B` on `devices`, each with its backend's default settings,
    /// apart from every other group, even one of the same devices.
    ///
    /// Make a group once and reuse it, as any device: each call registers a new group for the
    /// rest of the process.
    ///
    /// # Panics
    ///
    /// As [`with_member_settings`](Self::with_member_settings).
    pub fn new<B: BackendIr>(devices: &[B::Device]) -> Result<Self, GroupError> {
        let settings: Vec<DeviceSettings> = devices.iter().map(DeviceOps::defaults).collect();
        Self::with_member_settings::<B>(devices, &settings)
    }

    /// A new group whose members run `B` on `devices`, `settings` holding each one's own in member
    /// order. Each of the group's default dtypes is the first member's own that every member
    /// supports.
    ///
    /// # Panics
    ///
    /// When the process already has 65,536 groups.
    pub fn with_member_settings<B: BackendIr>(
        devices: &[B::Device],
        settings: &[DeviceSettings],
    ) -> Result<Self, GroupError> {
        if devices.is_empty() {
            return Err(GroupError::NoDevices);
        }
        if devices.len() != settings.len() {
            return Err(GroupError::SettingsPerMember {
                devices: devices.len(),
                settings: settings.len(),
            });
        }
        let interpreter = GroupInterpreter::<B>::new(devices.to_vec());
        let group_settings = interpreter.default_settings(settings)?;
        let index = NEXT_INDEX.fetch_add(1, Ordering::Relaxed);
        let device = Self {
            index: u16::try_from(index).expect("Fewer than 65536 device groups in one process"),
        };
        let client = GroupClient::new(device, Box::new(interpreter), group_settings);
        register_scoped_client::<GroupChannel>(&device, client)
            .expect("A new group's device has no client yet")
            .keep();
        Ok(device)
    }
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevices => write!(f, "A device group needs at least one device"),
            Self::MixedBackends => write!(f, "A device group's members run one backend"),
            Self::MixedRuntimes => write!(f, "A device group's members run one runtime"),
            Self::AutodiffMember => write!(
                f,
                "A device group is made from plain devices: group them, then make the group \
                 autodiff"
            ),
            Self::RemoteMember => write!(
                f,
                "Remote devices cannot form a device group: tensor parallelism needs links as \
                 fast as one machine's GPU interconnect"
            ),
            Self::UnsupportedBackend { device } => {
                write!(
                    f,
                    "A device group's members run on Cube or Flex, not {device}"
                )
            }
            Self::NoSharedDefault { candidates } => write!(
                f,
                "None of the members' defaults {candidates:?} runs on every member"
            ),
            Self::SettingsPerMember { devices, settings } => write!(
                f,
                "A device group takes one settings per member, not {settings} for {devices}"
            ),
        }
    }
}

impl std::error::Error for GroupError {}

/// A group exists only once made from its members, so there is no group to default to.
impl Default for GroupDevice {
    fn default() -> Self {
        panic!(
            "A device group is made from its devices, with Device::group_options or \
             GroupDevice::new"
        )
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
