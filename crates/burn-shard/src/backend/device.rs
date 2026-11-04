use std::{
    fmt,
    sync::atomic::{AtomicU32, Ordering},
};

use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, RouterDeviceType};
use burn_ir::BackendIr;
use burn_router::{get_client, register_scoped_client};
use burn_std::{QuantConfig, device::Device};

use super::{ShardClient, ShardCustomOps, channel::ShardChannel, interpreter::ShardInterpreter};

/// A sharded compute device of devices, its members, that a tensor's shards are spread over.
///
/// Like any device, a sharded compute device lasts as long as the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardedComputeDevice {
    /// Which sharded compute device of the process this is, in creation order, not a member's position.
    id: u16,
}

static NEXT_ID: AtomicU32 = AtomicU32::new(0);

/// Why a set of devices cannot form a sharded compute device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardError {
    /// A sharded compute device needs at least one member.
    NoDevices,
    /// Every member of a sharded compute device runs one backend.
    MixedBackends,
    /// A sharded compute device is made from plain devices: shard them, then make the result autodiff.
    AutodiffMember,
    /// Tensor parallelism needs links as fast as one machine's GPU interconnect, which a remote
    /// device does not have.
    RemoteMember,
    /// Only Cube and Flex devices form a sharded compute device through burn-dispatch.
    UnsupportedBackend {
        /// The device that cannot be a member.
        device: String,
    },
    /// No member's default for a dtype runs on every member.
    NoSharedDefault {
        /// Each member's default for that dtype, in member order.
        candidates: Vec<DType>,
    },
    /// Every member of a sharded compute device quantizes the same way, since one op quantizes on all of them.
    MixedQuantization {
        /// Each member's quantization config, in member order.
        configs: Vec<QuantConfig>,
    },
    /// The custom operations were registered for another backend than the members run.
    CustomOpsBackend {
        /// The backend the custom operations were registered for.
        registry: String,
        /// The backend the members run.
        members: String,
    },
}

/// A member of a sharded compute device: its device, and the settings it runs with.
#[derive(Clone, Debug)]
pub struct MemberDevice<D> {
    /// The member's device.
    pub device: D,
    /// The member's own settings.
    pub settings: DeviceSettings,
}

impl ShardedComputeDevice {
    /// A new sharded compute device whose members run `B` on `devices`, each with its backend's default settings,
    /// apart from every other one, even one of the same devices.
    ///
    /// Make a sharded compute device once and reuse it, as any device: each call registers a new sharded compute device for the
    /// rest of the process.
    ///
    /// # Errors
    ///
    /// As [`with_member_settings`](Self::with_member_settings).
    ///
    /// # Panics
    ///
    /// As [`with_member_settings`](Self::with_member_settings).
    pub fn new<B: BackendIr>(devices: &[B::Device]) -> Result<Self, ShardError> {
        let members: Vec<MemberDevice<B::Device>> = devices
            .iter()
            .map(|device| MemberDevice {
                device: device.clone(),
                settings: device.defaults(),
            })
            .collect();
        Self::with_member_settings::<B>(&members, ShardCustomOps::default())
    }

    /// A new sharded compute device whose `members` run `B`, each with its own settings and with
    /// `B`'s handlers in `custom_ops`. Each of its default dtypes is the first of the members' own,
    /// in member order, that every member supports.
    ///
    /// A custom operation without a [rule](crate::CustomOpRule) gathers its inputs and runs whole
    /// on every member.
    ///
    /// # Errors
    ///
    /// When `members` is empty, when no member's default for a dtype runs on every member, when
    /// the members quantize differently, or when `custom_ops` are another backend's.
    ///
    /// # Panics
    ///
    /// When the process already has 65,536 sharded compute devices.
    pub fn with_member_settings<B: BackendIr>(
        members: &[MemberDevice<B::Device>],
        custom_ops: ShardCustomOps,
    ) -> Result<Self, ShardError> {
        if members.is_empty() {
            return Err(ShardError::NoDevices);
        }
        let devices: Vec<B::Device> = members.iter().map(|it| it.device.clone()).collect();
        let interpreter = ShardInterpreter::<B>::new(devices.clone(), custom_ops)?;
        let shared_settings = interpreter.default_settings(members)?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let device = Self {
            id: u16::try_from(id).expect("Fewer than 65536 sharded compute devices in one process"),
        };
        let client = ShardClient::new(device, interpreter, shared_settings, devices);
        register_scoped_client::<ShardChannel>(&device, client)
            .expect("A new sharded compute device's device has no client yet")
            .keep();
        Ok(device)
    }

    /// The devices of its members, in member order, when they run `B`, or `None` when they run
    /// another backend.
    pub fn members<B: BackendIr>(&self) -> Option<Vec<B::Device>> {
        get_client::<ShardChannel>(self).members::<B::Device>()
    }
}

impl fmt::Display for ShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevices => write!(f, "A sharded compute device needs at least one device"),
            Self::MixedBackends => write!(f, "A sharded compute device's members run one backend"),
            Self::AutodiffMember => write!(
                f,
                "A sharded compute device is made from plain devices: shard them, then make the result \
                 autodiff"
            ),
            Self::RemoteMember => write!(
                f,
                "Remote devices cannot form a sharded compute device: tensor parallelism needs links as \
                 fast as one machine's GPU interconnect"
            ),
            Self::UnsupportedBackend { device } => {
                write!(
                    f,
                    "A sharded compute device's members run on Cube or Flex, not {device}"
                )
            }
            Self::NoSharedDefault { candidates } => write!(
                f,
                "None of the members' defaults {candidates:?} runs on every member"
            ),
            Self::MixedQuantization { configs } => write!(
                f,
                "A sharded compute device's members quantize the same way, not as {configs:?}"
            ),
            Self::CustomOpsBackend { registry, members } => write!(
                f,
                "The custom operations are registered for {registry}, and the members run \
                 {members}"
            ),
        }
    }
}

impl std::error::Error for ShardError {}

/// A sharded compute device exists only once made from its members, so there is no sharded compute device to default to.
impl Default for ShardedComputeDevice {
    fn default() -> Self {
        panic!(
            "A sharded compute device is made from its devices, with Device::shard_options or \
             ShardedComputeDevice::new"
        )
    }
}

impl Device for ShardedComputeDevice {
    fn from_id(device_id: DeviceId) -> Self {
        assert_eq!(
            device_id.type_id,
            u16::from(RouterDeviceType::ShardedCompute),
            "Not a sharded compute device id"
        );
        Self {
            id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: RouterDeviceType::ShardedCompute.into(),
            index_id: self.id,
        }
    }
}

impl DeviceOps for ShardedComputeDevice {
    fn defaults(&self) -> DeviceSettings {
        get_client::<ShardChannel>(self).settings()
    }
}
