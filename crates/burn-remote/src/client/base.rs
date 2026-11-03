use super::{ConnectError, RemoteDevice, service::RemoteService};
use burn_backend::{DeviceHandle, backend::Device};
use burn_std::rand::StdRng;
use std::sync::{Arc, Mutex};

/// A thin handle to a `RemoteService` running on its own device-runner thread.
///
/// Connection state, the tokio runtime, the response-demux task, and the op batch buffer
/// live on the service side. Random keys are reserved locally from the device's shared generator.
pub struct RemoteClient {
    pub(crate) device: RemoteDevice,
    pub(crate) handle: DeviceHandle<RemoteService>,
    pub(crate) rng: Arc<Mutex<StdRng>>,
}

impl RemoteClient {
    pub fn init(device: RemoteDevice) -> Self {
        // `DeviceHandle::new` initializes the service the first time it's called for a given
        // device id. `RemoteService::init` is deliberately cheap — it records the endpoint but
        // does NOT connect, because cubecl holds a process-global lock across it; the actual
        // connect + handshake happens lazily on first use (or via `connect`).
        // Subsequent calls return a handle to the existing service.
        let handle = DeviceHandle::<RemoteService>::new(device.to_id());
        let rng = super::service::random_generator(device.id);
        Self {
            device,
            handle,
            rng,
        }
    }

    /// Open the lazily-established connection now, populating the device's settings/device-count
    /// cells, or return why it could not be opened. Runs the connect on the service's runner
    /// thread, so it can't sit under cubecl's global lock.
    pub(crate) fn connect(&self) -> Result<(), ConnectError> {
        self.handle
            .submit_blocking(|s| s.try_connect())
            .expect("Service call failed")
    }

    /// Establish the session asynchronously, the way the browser requires.
    ///
    /// The connect + handshake cannot block the single browser thread, so it runs off the device
    /// handle: the service hands back the connection parameters, the network round-trip happens
    /// with `.await`, and the opened session is installed back into the service. A no-op once the
    /// session is up.
    #[cfg(target_family = "wasm")]
    pub(crate) async fn connect_async(&self) -> Result<(), ConnectError> {
        use crate::client::service::wasm_connect;

        let Some(plan) = self
            .handle
            .submit_blocking(|s| s.wasm_connect_plan())
            .expect("Service call failed")
        else {
            return Ok(());
        };

        let connected = wasm_connect(plan).await?;

        self.handle
            .submit_blocking(move |s| s.wasm_install(connected))
            .expect("Service call failed");
        Ok(())
    }
}

impl Clone for RemoteClient {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            handle: self.handle.clone(),
            rng: self.rng.clone(),
        }
    }
}

#[cfg(all(test, feature = "websocket", not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::{Credential, client::service::RemoteEndpoint};
    use burn_communication::Address;
    use burn_router::RouterClient;
    use std::{sync::mpsc, thread, time::Duration};

    #[test]
    fn clients_share_random_keys_without_waiting_for_the_service() {
        let device = RemoteDevice::register(
            RemoteEndpoint::WebSocket {
                address: Address::from("ws://127.0.0.1:16371"),
                credential: Credential::default(),
            },
            0,
        );
        let client = RemoteClient::init(device.clone());
        let cloned = client.clone();
        let other = RemoteClient::init(device);
        client.seed(7);
        let expected: [_; 3] = core::array::from_fn(|_| client.next_random_key());

        let timeout = Duration::from_secs(5);
        let (release_tx, release_rx) = mpsc::channel();
        client.handle.submit(move |_| {
            let _ = release_rx.recv();
        });
        // Queue the blocker before the caller can submit a key request to this worker.
        client.handle.flush_queue();

        let (keys_tx, keys_rx) = mpsc::channel();
        let caller = thread::spawn(move || {
            other.seed(7);
            keys_tx
                .send([
                    client.next_random_key(),
                    cloned.next_random_key(),
                    other.next_random_key(),
                ])
                .unwrap();
        });
        let keys = keys_rx.recv_timeout(timeout);
        // Release the service even on failure so a blocking implementation can finish.
        release_tx.send(()).unwrap();
        caller.join().unwrap();
        assert_eq!(
            keys.expect("random keys must not wait for the service"),
            expected
        );
    }
}
