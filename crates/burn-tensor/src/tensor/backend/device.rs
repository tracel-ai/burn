pub use burn_std::device::*;

/// The handle device trait allows to get an id for a backend device.
pub trait DeviceOps:
    Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + burn_std::device::Device
{
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }
}
