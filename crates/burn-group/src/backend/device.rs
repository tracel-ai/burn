use std::{
    fmt,
    sync::atomic::{AtomicU32, Ordering},
};

use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, RouterDeviceType};
use burn_ir::BackendIr;
use burn_router::{get_client, register_scoped_client};
use burn_std::{QuantConfig, device::Device};

use super::{GroupClient, channel::GroupChannel, interpreter::GroupInterpreter};

/// A group of devices, its members, that a tensor's shards are spread over.
///
/// Like any device, a group lasts as long as the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupDevice {
    /// Which group of the process this is, in creation order, not a member's position.
    id: u16,
}

static NEXT_ID: AtomicU32 = AtomicU32::new(0);

/// Why a set of devices cannot form a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupError {
    /// A group needs at least one member.
    NoDevices,
    /// Every member of a group runs one backend.
    MixedBackends,
    /// A group is made from plain devices: group them, then make the group autodiff.
    AutodiffMember,
    /// Tensor parallelism needs links as fast as one machine's GPU interconnect, which a remote
    /// device does not have.
    RemoteMember,
    /// Only Cube and Flex devices form a group through burn-dispatch.
    UnsupportedBackend {
        /// The device that cannot be a member.
        device: String,
    },
    /// No member's default for a dtype runs on every member.
    NoSharedDefault {
        /// Each member's default for that dtype, in member order.
        candidates: Vec<DType>,
    },
    /// Every member of a group quantizes the same way, since one op quantizes on all of them.
    MixedQuantization {
        /// Each member's quantization config, in member order.
        configs: Vec<QuantConfig>,
    },
}

/// A member of a group: its device, and the settings it runs with.
#[derive(Clone, Debug)]
pub struct MemberDevice<D> {
    /// The member's device.
    pub device: D,
    /// The member's own settings.
    pub settings: DeviceSettings,
}

impl GroupDevice {
    /// A new group whose members run `B` on `devices`, each with its backend's default settings,
    /// apart from every other group, even one of the same devices.
    ///
    /// Make a group once and reuse it, as any device: each call registers a new group for the
    /// rest of the process.
    ///
    /// # Errors
    ///
    /// As [`with_member_settings`](Self::with_member_settings).
    ///
    /// # Panics
    ///
    /// As [`with_member_settings`](Self::with_member_settings).
    pub fn new<B: BackendIr>(devices: &[B::Device]) -> Result<Self, GroupError> {
        let members: Vec<MemberDevice<B::Device>> = devices
            .iter()
            .map(|device| MemberDevice {
                device: device.clone(),
                settings: device.defaults(),
            })
            .collect();
        Self::with_member_settings::<B>(&members)
    }

    /// A new group whose `members` run `B`, each with its own settings. Each of the group's
    /// default dtypes is the first of the members' own, in member order, that every member
    /// supports.
    ///
    /// # Errors
    ///
    /// When `members` is empty, when no member's default for a dtype runs on every member, or
    /// when the members quantize differently.
    ///
    /// # Panics
    ///
    /// When the process already has 65,536 groups.
    pub fn with_member_settings<B: BackendIr>(
        members: &[MemberDevice<B::Device>],
    ) -> Result<Self, GroupError> {
        if members.is_empty() {
            return Err(GroupError::NoDevices);
        }
        let interpreter =
            GroupInterpreter::<B>::new(members.iter().map(|it| it.device.clone()).collect());
        let group_settings = interpreter.default_settings(members)?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let device = Self {
            id: u16::try_from(id).expect("Fewer than 65536 device groups in one process"),
        };
        let client = GroupClient::new(device, interpreter, group_settings);
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
            Self::MixedQuantization { configs } => write!(
                f,
                "A device group's members quantize the same way, not as {configs:?}"
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
            id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: RouterDeviceType::Group.into(),
            index_id: self.id,
        }
    }
}

impl DeviceOps for GroupDevice {
    fn defaults(&self) -> DeviceSettings {
        get_client::<GroupChannel>(self).settings()
    }
}
