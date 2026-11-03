pub use burn_backend::{EnvironmentBundle, EnvironmentError};

use crate::Device;
use alloc::string::String;
use burn_backend::{Backend, EnvironmentId, EnvironmentOrigin};
use burn_dispatch::Dispatch;

/// A compute environment: the compiled kernels and autotune answers the work
/// of the devices it is active on reads and writes, so a later run reuses them
/// instead of paying for them again.
///
/// It lives in the storage of the device that created it — a process's cache
/// directory, or a remote device's server — and can be
/// [activated](Self::activate) on any device that reaches that storage. Entries
/// are kept apart by the hardware they were made for: devices of one
/// architecture reuse each other's compiled kernels, devices of one model each
/// other's autotune answers, and unlike devices share an environment without
/// reading each other's.
///
/// ```rust,ignore
/// // A build, kept apart from the environment the application runs in.
/// let previous = ComputeEnvironment::active(&device)?;
/// let built = ComputeEnvironment::new(&device, "building")?;
/// built.activate(&device)?;
/// // ... compile-only and tune passes ...
/// std::fs::write(path, &*built.save()?.bytes)?;
/// previous.activate(&device)?;
///
/// // Later, or elsewhere: run in what the build kept.
/// let bundle = EnvironmentBundle { bytes: Bytes::from_bytes_vec(std::fs::read(path)?) };
/// ComputeEnvironment::load(&device, "model", bundle)?.activate(&device)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ComputeEnvironment {
    /// A device that reaches the storage the environment lives in.
    home: Device,
    id: EnvironmentId,
}

impl ComputeEnvironment {
    /// Create the empty compute environment `name` in the storage `device`
    /// reaches, replacing one of that name there.
    ///
    /// # Errors
    ///
    /// The device keeps no compute environment, the one to replace is active,
    /// or the storage refused the write.
    pub fn new(device: &Device, name: impl Into<String>) -> Result<Self, EnvironmentError> {
        Self::create(device, name.into(), EnvironmentOrigin::Empty)
    }

    /// Create the compute environment `name` in the storage `device` reaches,
    /// holding what `bundle` holds, replacing one of that name there.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), and when the bundle does not read.
    pub fn load(
        device: &Device,
        name: impl Into<String>,
        bundle: EnvironmentBundle,
    ) -> Result<Self, EnvironmentError> {
        Self::create(device, name.into(), EnvironmentOrigin::Bundle(bundle))
    }

    /// Create the compute environment `name` beside this one, holding a copy
    /// of what this one holds now.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn fork(&self, name: impl Into<String>) -> Result<Self, EnvironmentError> {
        Self::create(
            &self.home,
            name.into(),
            EnvironmentOrigin::Fork(self.id.clone()),
        )
    }

    /// The compute environment `device`'s work reads and writes now.
    ///
    /// # Errors
    ///
    /// The device keeps no compute environment.
    pub fn active(device: &Device) -> Result<Self, EnvironmentError> {
        let id = Dispatch::environment_active(device.as_dispatch())?;
        Ok(Self {
            home: device.clone(),
            id,
        })
    }

    /// Make this the compute environment `device`'s work reads and writes,
    /// after every operation registered so far on the calling stream, and
    /// answer the one it replaces.
    ///
    /// At least `device` works in it. The cubecl runtimes keep one active
    /// environment per process, so there every device of the process does —
    /// on a remote device, every client of its server.
    ///
    /// # Errors
    ///
    /// [`IncompatibleDevice`](EnvironmentError::IncompatibleDevice) when
    /// `device` does not reach the storage this environment lives in: carry
    /// it there by [saving](Self::save) it and [loading](Self::load) the
    /// bundle on `device`.
    pub fn activate(&self, device: &Device) -> Result<Self, EnvironmentError> {
        let previous = Dispatch::environment_activate(device.as_dispatch(), self.id.clone())?;
        Ok(Self {
            home: device.clone(),
            id: previous,
        })
    }

    /// Serialize the environment whole, every device's entries in it, after
    /// every operation registered so far on the calling stream.
    ///
    /// # Errors
    ///
    /// The storage refused the read, or the device could not be reached.
    pub fn save(&self) -> Result<EnvironmentBundle, EnvironmentError> {
        Dispatch::environment_save(self.home.as_dispatch(), self.id.clone())
    }

    /// The environment's name in the storage it lives in.
    pub fn name(&self) -> &str {
        &self.id.name
    }

    fn create(
        device: &Device,
        name: String,
        origin: EnvironmentOrigin,
    ) -> Result<Self, EnvironmentError> {
        let id = Dispatch::environment_create(device.as_dispatch(), name, origin)?;
        Ok(Self {
            home: device.clone(),
            id,
        })
    }
}
