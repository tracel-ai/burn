use core::cmp::Ordering;

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
}

/// The handle device trait allows to get an id for a backend device.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug {
    /// Return the [device id](DeviceId).
    fn id(&self) -> DeviceId;
}

impl core::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl Ord for DeviceId {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.type_id.cmp(&other.type_id) {
            Ordering::Equal => self.index_id.cmp(&other.index_id),
            other => other,
        }
    }
}

impl PartialOrd for DeviceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
