pub use burn_std::device::*;

/// Device trait for all burn backend devices.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + Device {
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }

    // TODO: is this required? Might want to rethink about the InnerBackend / device association with Dispatch

    /// Returns the inner device.
    fn inner(&self) -> &Self {
        self
    }
}
