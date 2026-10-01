//! What a client connects to, and the connect itself.

use core::future::Future;

use burn_router::get_client;

use super::service::RemoteEndpoint;
use super::{ConnectError, RemoteChannel, RemoteDevice, service};
use crate::Credential;
#[cfg(feature = "iroh")]
use crate::transport::iroh::IrohHost;
#[cfg(feature = "websocket")]
use burn_communication::Address;

/// A remote server as a client reaches it: where it is, over which transport, and the credential
/// to present. Building one opens nothing.
///
/// The facade over it is `burn::remote::RemoteHost`, which turns the devices this connects into
/// Burn devices.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HostSpec {
    target: Target,
    credential: Credential,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Target {
    #[cfg(feature = "websocket")]
    WebSocket(Address),
    #[cfg(feature = "iroh")]
    Iroh(IrohHost),
}

impl HostSpec {
    /// The WebSocket server at `url`, such as `ws://gpu:3000`.
    #[cfg(feature = "websocket")]
    pub fn websocket(url: &str) -> Self {
        Self {
            target: Target::WebSocket(Address::from(url)),
            credential: Credential::default(),
        }
    }

    /// The Iroh server `host` describes.
    #[cfg(feature = "iroh")]
    pub fn iroh(host: impl Into<IrohHost>) -> Self {
        Self {
            target: Target::Iroh(host.into()),
            credential: Credential::default(),
        }
    }

    /// What the server's authorizer checks.
    pub fn with_credential(mut self, credential: impl Into<Credential>) -> Self {
        self.credential = credential.into();
        self
    }

    /// Open a session to device `index` and wait for its handshake.
    ///
    /// # Errors
    ///
    /// See [`ConnectError`].
    #[cfg(not(target_family = "wasm"))]
    pub fn connect(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        self.refuse_blocking_on_current_thread()?;
        let host = self.clone();
        on_burn_runtime::wait(move || host.connect_blocking(index))
    }

    /// Open a session to device `index`. Dropping the future does not cancel a connect that has
    /// started: it finishes in the background.
    ///
    /// # Errors
    ///
    /// See [`ConnectError`].
    #[cfg(not(target_family = "wasm"))]
    pub fn connect_async(
        &self,
        index: usize,
    ) -> impl Future<Output = Result<RemoteDevice, ConnectError>> + Send + 'static + use<> {
        let host = self.clone();
        on_burn_runtime::run(move || host.connect_blocking(index))
    }

    /// Open a session to device `index`.
    ///
    /// # Errors
    ///
    /// See [`ConnectError`].
    #[cfg(target_family = "wasm")]
    pub fn connect_async(
        &self,
        index: usize,
    ) -> impl Future<Output = Result<RemoteDevice, ConnectError>> + 'static + use<> {
        let host = self.clone();
        async move { host.connect_in_browser(index).await }
    }

    /// Open a session to every device the server hosts.
    ///
    /// # Errors
    ///
    /// The first device that cannot be connected, refusals included, fails the whole list.
    #[cfg(not(target_family = "wasm"))]
    pub fn devices(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        self.refuse_blocking_on_current_thread()?;
        let host = self.clone();
        on_burn_runtime::wait(move || host.devices_blocking())
    }

    /// Asynchronous [`devices`](Self::devices).
    ///
    /// # Errors
    ///
    /// The first device that cannot be connected, refusals included, fails the whole list.
    #[cfg(not(target_family = "wasm"))]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Vec<RemoteDevice>, ConnectError>> + Send + 'static + use<>
    {
        let host = self.clone();
        on_burn_runtime::run(move || host.devices_blocking())
    }

    /// Open a session to every device the server hosts.
    ///
    /// # Errors
    ///
    /// The first device that cannot be connected, refusals included, fails the whole list.
    #[cfg(target_family = "wasm")]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Vec<RemoteDevice>, ConnectError>> + 'static + use<> {
        let host = self.clone();
        async move {
            let first = host.connect_in_browser(0).await?;
            let mut devices = vec![first];
            for index in 1..host_device_count(&devices[0]) {
                devices.push(host.connect_in_browser(index).await?);
            }
            Ok(devices)
        }
    }

    async fn endpoint(&self) -> Result<RemoteEndpoint, ConnectError> {
        match &self.target {
            #[cfg(feature = "websocket")]
            Target::WebSocket(address) => Ok(RemoteEndpoint::WebSocket {
                address: address.clone(),
                credential: self.credential.clone(),
            }),
            #[cfg(feature = "iroh")]
            Target::Iroh(host) => {
                host.validate()?;
                Ok(RemoteEndpoint::Iroh {
                    node: host.node().await?,
                    peer: host.dial_addr(),
                    credential: self.credential.clone(),
                    app_endpoint: host.app_endpoint(),
                })
            }
        }
    }

    /// Runs on a blocking thread of Burn's runtime, never on one of its workers: the connect
    /// blocks the device's runner, and enough runners blocking every worker would stop all I/O.
    #[cfg(not(target_family = "wasm"))]
    fn connect_blocking(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        let endpoint = super::runtime::blocking_runtime()
            .handle()
            .block_on(self.endpoint())?;
        let device = RemoteDevice::register(endpoint, index);
        get_client::<RemoteChannel>(&device).connect()?;
        Ok(device)
    }

    #[cfg(not(target_family = "wasm"))]
    fn devices_blocking(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        let first = self.connect_blocking(0)?;
        let count = host_device_count(&first);
        let mut devices = vec![first];
        for index in 1..count {
            devices.push(self.connect_blocking(index)?);
        }
        Ok(devices)
    }

    #[cfg(target_family = "wasm")]
    async fn connect_in_browser(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        let endpoint = self.endpoint().await?;
        let device = RemoteDevice::register(endpoint, index);
        get_client::<RemoteChannel>(&device).connect_async().await?;
        Ok(device)
    }

    /// An application endpoint runs its socket on the runtime that bound it. Blocking the only
    /// thread of a current-thread runtime would starve it, and the connect would never finish.
    #[cfg(not(target_family = "wasm"))]
    fn refuse_blocking_on_current_thread(&self) -> Result<(), ConnectError> {
        #[cfg(feature = "iroh")]
        if let Target::Iroh(host) = &self.target
            && host.app_endpoint().is_some()
            && let Ok(handle) = tokio::runtime::Handle::try_current()
            && handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread
        {
            return Err(ConnectError::InvalidConfiguration {
                reason: "a blocking connect on a current-thread runtime cannot drive the \
                         application endpoint that runtime runs; use `init_async`"
                    .into(),
            });
        }
        Ok(())
    }
}

fn host_device_count(device: &RemoteDevice) -> usize {
    service::device_count_for(device.id).expect("the handshake reports the device count") as usize
}

/// Work run on a blocking thread of Burn's runtime, and waited on from any thread or executor.
#[cfg(not(target_family = "wasm"))]
mod on_burn_runtime {
    use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

    use crate::client::runtime::blocking_runtime;

    /// Block the calling thread until `work` finishes. A plain channel, so a caller on a runtime
    /// thread blocks it rather than panicking.
    pub(super) fn wait<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
        let (sender, receiver) = std::sync::mpsc::channel();
        blocking_runtime().spawn_blocking(move || {
            let _ = sender.send(catch_unwind(AssertUnwindSafe(work)));
        });
        match receiver.recv() {
            Ok(Ok(value)) => value,
            Ok(Err(panic)) => resume_unwind(panic),
            Err(_) => panic!("Burn Remote's runtime dropped a connect"),
        }
    }

    /// Await `work` from any executor.
    pub(super) fn run<T: Send + 'static>(
        work: impl FnOnce() -> T + Send + 'static,
    ) -> impl core::future::Future<Output = T> + Send + 'static {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        blocking_runtime().spawn_blocking(move || {
            let _ = sender.send(catch_unwind(AssertUnwindSafe(work)));
        });
        async move {
            match receiver.await {
                Ok(Ok(value)) => value,
                Ok(Err(panic)) => resume_unwind(panic),
                Err(_) => panic!("Burn Remote's runtime dropped a connect"),
            }
        }
    }
}
