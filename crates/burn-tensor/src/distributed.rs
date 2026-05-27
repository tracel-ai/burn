//! Distributed execution utilities.
//!
//! The core component of this module is [`DistributedContext`], which manages
//! the lifecycle of distributed synchronization clients.

use alloc::vec::Vec;
use burn_backend::distributed::DistributedBackend;
use burn_dispatch::Dispatch;
pub use burn_std::distributed::DistributedConfig;

use crate::Device;

/// This structure acts as a resource handle for multi-device synchronization.
///
/// Spawning this context automatically initializes the underlying distributed communication
/// servers, while dropping it guarantees a clean and safe teardown of all network resources.
#[derive(Debug)]
pub struct DistributedContext {
    devices: Vec<Device>,
}

impl DistributedContext {
    /// Starts a distributed communication server for the provided devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The collection of compute devices participating in the distributed operations.
    /// * `config` - Parameter aggregation settings, such as global reduction strategies (`Mean`, `Sum`, etc.).
    pub fn init(devices: Vec<Device>, config: DistributedConfig) -> Self {
        let dispatch_devices = devices
            .iter()
            .map(|d| d.as_dispatch().clone())
            .collect::<Vec<_>>();
        Dispatch::start_communication_server(&dispatch_devices, config);

        Self { devices }
    }
}

impl Drop for DistributedContext {
    fn drop(&mut self) {
        if !self.devices.is_empty() {
            Dispatch::close_communication_server(self.devices[0].as_dispatch());
        }
    }
}
