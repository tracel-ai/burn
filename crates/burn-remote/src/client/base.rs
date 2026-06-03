use super::{RemoteDevice, service::RemoteService};
use burn_backend::{DeviceHandle, backend::Device};
use burn_communication::ProtocolClient;

/// A thin handle to a [`RemoteService`] running on its own device-runner thread.
///
/// Every `RouterClient` method delegates to `handle.submit` / `submit_blocking`, so all
/// connection state, the tokio runtime, the response-demux task, and the op batch buffer
/// live on the service side.
pub struct RemoteClient<C: ProtocolClient> {
    pub(crate) device: RemoteDevice,
    pub(crate) handle: DeviceHandle<RemoteService<C>>,
}

impl<C: ProtocolClient> RemoteClient<C> {
    pub fn init(device: RemoteDevice) -> Self {
        // `DeviceHandle::new` is the path that initializes the service the first time it's
        // called for a given device id — `RemoteService::init` then connects and runs the
        // handshake. Subsequent calls return a handle to the existing service.
        let handle = DeviceHandle::<RemoteService<C>>::new(device.to_id());
        Self { device, handle }
    }
}

impl<C: ProtocolClient> Clone for RemoteClient<C> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            handle: self.handle.clone(),
        }
    }
}
