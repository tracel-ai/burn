//! Distributed execution utilities.
//!
//! The core component of this module is [`DistributedContext`], which manages
//! the lifecycle of distributed synchronization clients.

use alloc::vec::Vec;
use burn_backend::TensorMetadata;
use burn_backend::{DeviceOps, distributed::DistributedOps};
use burn_dispatch::{Dispatch, DispatchTensor};
pub use burn_std::distributed::*;

use crate::{Device, Tensor, ops::BridgeTensor};

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

/// A tensor handle used for a collective operation, that is not yet valid for use.
/// We must ensure collective operations are completed before accessing the underlying data.
#[derive(new, Clone)]
pub struct CollectiveTensor<const D: usize> {
    handle: DispatchTensor,
}

impl<const D: usize> CollectiveTensor<D> {
    /// Synchronizes the collective operation and returns a valid tensor handle.
    pub fn resolve(self) -> Tensor<D> {
        Dispatch::sync_collective(&self.handle.device());
        Tensor::new(BridgeTensor::float(self.handle))
    }

    /// Returns the tensor handle without synchronizing.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `sync_collective()` is called before
    /// the returned handle is used in any computation.
    pub unsafe fn assume_resolved(self) -> Tensor<D> {
        Tensor::new(BridgeTensor::float(self.handle))
    }
}

/// Performs an all_reduce operation on the input tensor.
///
/// # Arguments
/// - `input`: The input tensor.
/// - `op`: The aggregation operation.
/// - `device_ids`: The list of all devices with which to `all_reduce`
///
/// # Returns
/// A [CollectiveTensor] containing the handle of the result.
pub fn all_reduce<const D: usize>(
    input: Tensor<D>,
    op: ReduceOperation,
    device_ids: Vec<Device>,
) -> CollectiveTensor<D> {
    let device_ids = device_ids.iter().map(|d| d.as_dispatch().id()).collect();
    let collective = Dispatch::all_reduce(input.primitive.into_float(), op, device_ids);
    // Safety: we call `assume_resolved` only to wrap it in `burn_tensor`'s [CollectiveTensor].
    CollectiveTensor::new(unsafe { collective.assume_resolved() })
}
