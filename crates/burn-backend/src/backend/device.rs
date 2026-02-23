pub use burn_std::device::*;

/// Device trait for all burn backend devices.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + Device {
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }

    /// Returns the inner device without autodiff enabled.
    ///
    /// For most devices this is a no-op that returns `self`. For autodiff-enabled
    /// devices, this returns the underlying inner device.
    fn inner(&self) -> &Self {
        self
    }
}
