//! What a client connects to, and the connect itself.

use core::future::Future;

use super::service::RemoteEndpoint;
use super::{ConnectError, RemoteDevice, service};
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
        let endpoint = self.endpoint_blocking()?;
        let device = RemoteDevice::open(endpoint.clone(), index)?;
        if device.session_ended() {
            // A new registration has no session to confirm, so it opens one.
            return RemoteDevice::open(endpoint, index);
        }
        Ok(device)
    }

    /// Device 0 connects for the count; the others connect on first use, as a local backend's
    /// listed devices initialize on first use.
    #[cfg(not(target_family = "wasm"))]
    fn devices_blocking(&self) -> Result<Vec<RemoteDevice>, ConnectError> {
        let first = self.connect_blocking(0)?;
        let count = host_device_count(&first);
        let endpoint = self.endpoint_blocking()?;
        let others = (1..count).map(|index| RemoteDevice::register(endpoint.clone(), index));
        Ok(core::iter::once(first).chain(others).collect())
    }

    #[cfg(not(target_family = "wasm"))]
    fn endpoint_blocking(&self) -> Result<RemoteEndpoint, ConnectError> {
        runtime::blocking_runtime()
            .handle()
            .block_on(self.endpoint())
    }

    #[cfg(target_family = "wasm")]
    async fn connect_in_browser(&self, index: usize) -> Result<RemoteDevice, ConnectError> {
        let endpoint = self.endpoint().await?;
        let device = RemoteDevice::open_async(endpoint.clone(), index).await?;
        if device.session_ended() {
            // A new registration has no session to confirm, so it opens one.
            return RemoteDevice::open_async(endpoint, index).await;
        }
        Ok(device)
    }
}

fn host_device_count(device: &RemoteDevice) -> usize {
    service::device_count_for(device.id).expect("the handshake reports the device count") as usize
}
