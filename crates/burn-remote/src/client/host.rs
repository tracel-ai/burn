//! What a client connects to, and the connect itself.

use core::future::Future;

use burn_router::get_client;

use super::service::RemoteEndpoint;
use super::{ConnectError, RemoteChannel, RemoteDevice, service};
use crate::Credential;
#[cfg(not(target_family = "wasm"))]
use crate::runtime;
#[cfg(feature = "iroh")]
use crate::transport::iroh::IrohHost;
#[cfg(feature = "websocket")]
use burn_communication::Address;

/// What `burn::remote::RemoteHost` wraps, which documents it.
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
    #[cfg(feature = "websocket")]
    pub fn websocket(url: &str) -> Self {
        Self {
            target: Target::WebSocket(Address::from(url)),
            credential: Credential::default(),
        }
    }

    #[cfg(feature = "iroh")]
    pub fn iroh(host: impl Into<IrohHost>) -> Self {
        Self {
            target: Target::Iroh(host.into()),
            credential: Credential::default(),
        }
    }

    pub fn with_credential(mut self, credential: impl Into<Credential>) -> Self {
        self.credential = credential.into();
        self
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn connect(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        self.refuse_blocking_on_current_thread()?;
        let host = self.clone();
        runtime::wait(move || host.connect_blocking(index))?
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn connect_async(
        &self,
        index: usize,
    ) -> impl Future<Output = Result<RemoteDevice, ConnectError>> + Send + 'static + use<> {
        let host = self.clone();
        async move { runtime::run(move || host.connect_blocking(index)).await? }
    }

    #[cfg(target_family = "wasm")]
    pub fn connect_async(
        &self,
        index: usize,
    ) -> impl Future<Output = Result<RemoteDevice, ConnectError>> + 'static + use<> {
        let host = self.clone();
        async move { host.connect_in_browser(index).await }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn devices(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        self.refuse_blocking_on_current_thread()?;
        let host = self.clone();
        runtime::wait(move || host.devices_blocking())?
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Vec<RemoteDevice>, ConnectError>> + Send + 'static + use<>
    {
        let host = self.clone();
        async move { runtime::run(move || host.devices_blocking()).await? }
    }

    #[cfg(target_family = "wasm")]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Vec<RemoteDevice>, ConnectError>> + 'static + use<> {
        let host = self.clone();
        async move {
            let first = host.connect_in_browser(0).await?;
            let others = (1..host_device_count(&first)).map(|index| host.connect_in_browser(index));
            let others = futures_util::future::try_join_all(others).await?;
            Ok(core::iter::once(first).chain(others).collect())
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

    /// Never runs on a worker of Burn's runtime: the connect blocks the device's runner, and
    /// enough runners blocking every worker would stop all I/O.
    #[cfg(not(target_family = "wasm"))]
    fn connect_blocking(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        let endpoint = runtime::blocking_runtime()
            .handle()
            .block_on(self.endpoint())?;
        let device = RemoteDevice::register(endpoint, index);
        get_client::<RemoteChannel>(&device).connect()?;
        Ok(device)
    }

    /// Device 0 first, for the count, then the others at once.
    #[cfg(not(target_family = "wasm"))]
    fn devices_blocking(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        let first = self.connect_blocking(0)?;
        let others = std::thread::scope(|scope| {
            let connects: Vec<_> = (1..host_device_count(&first))
                .map(|index| scope.spawn(move || self.connect_blocking(index)))
                .collect();
            connects
                .into_iter()
                .map(|connect| {
                    connect
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        Ok(core::iter::once(first).chain(others).collect())
    }

    #[cfg(target_family = "wasm")]
    async fn connect_in_browser(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        let endpoint = self.endpoint().await?;
        let device = RemoteDevice::register(endpoint, index);
        get_client::<RemoteChannel>(&device).connect_async().await?;
        Ok(device)
    }

    /// An application endpoint runs its socket on the runtime that bound it, which Iroh does not
    /// expose, so this only catches the visible case: a blocking connect inside a current-thread
    /// runtime.
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
                         application endpoint that runtime runs; use `init_async` or \
                         `devices_async`"
                    .into(),
            });
        }
        Ok(())
    }
}

fn host_device_count(device: &RemoteDevice) -> usize {
    service::device_count_for(device.id).expect("the handshake reports the device count") as usize
}
