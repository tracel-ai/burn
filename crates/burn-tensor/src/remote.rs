//! Reaching a remote compute server as a client.
//!
//! ```rust,ignore
//! let host = RemoteHost::iroh(server_id).with_credential(token);
//! let device = Device::remote_options(&host).init()?;
//! ```
//!
//! On native, every session runs on Burn's own runtime, so `init_async` and `devices_async`
//! can be awaited from any executor, and the blocking forms can be called from any thread. The
//! exception is an application endpoint bound on a current-thread runtime: any blocking call on a
//! device dialed from it, a read included, starves that runtime's thread, so bind such an
//! endpoint on a multi-thread runtime.

use core::future::Future;

use burn_dispatch::__remote::HostSpec;
pub use burn_dispatch::__remote::{
    ConnectError, Credential, CustomOpClient, Endpoint, EndpointAddr, EndpointId, InvalidRelays,
    IrohHost, IrohIdentity, IrohRelays, RelayUrl,
};

use crate::{Device, DeviceIndex, Devices};

/// A remote server as a client reaches it: where it is, over which transport, and the credential
/// its authorizer checks. Building one opens nothing; [`Device::remote_options`] connects a device
/// on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemoteHost(HostSpec);

impl RemoteHost {
    /// The WebSocket server at `url`, such as `ws://gpu:3000`.
    ///
    /// WebSocket is unencrypted: on it, a credential stops stray clients on a trusted network, not
    /// someone reading the traffic.
    #[cfg(feature = "remote-websocket")]
    pub fn websocket(url: &str) -> Self {
        Self(HostSpec::websocket(url))
    }

    /// The Iroh server `host` describes: its id, an [`EndpointAddr`] that also lists addresses,
    /// or an [`IrohHost`] with relay or endpoint settings.
    pub fn iroh(host: impl Into<IrohHost>) -> Self {
        Self(HostSpec::iroh(host))
    }

    /// What the server's authorizer checks, such as the token of a `TokenAuthorizer`.
    pub fn with_credential(self, credential: impl Into<Credential>) -> Self {
        Self(self.0.with_credential(credential))
    }

    /// Every device the server hosts. Device 0 connects, which reports the count; the others
    /// connect on first use, as a local backend's listed devices initialize on first use.
    /// `Device::enumerate(DeviceType::Remote(host))` lists the same devices, and panics where this
    /// returns an error.
    ///
    /// # Errors
    ///
    /// Device 0 cannot be connected. A device the authorizer refuses fails on first use instead,
    /// so a client allowed only some of the server's devices connects each with
    /// [`RemoteOptions::device_index`].
    #[cfg(not(target_family = "wasm"))]
    pub fn devices(&self) -> Result<Devices, ConnectError> {
        Ok(self.0.devices()?.into_iter().map(Device::new).collect())
    }

    /// Asynchronous [`devices`](Self::devices).
    #[cfg(not(target_family = "wasm"))]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Devices, ConnectError>> + Send + 'static + use<> {
        let devices = self.0.devices_async();
        async move { Ok(devices.await?.into_iter().map(Device::new).collect()) }
    }

    /// Connect every device the server hosts, one session each, since a browser cannot connect a
    /// device on first use. The first device that cannot be connected fails the whole list.
    #[cfg(target_family = "wasm")]
    pub fn devices_async(
        &self,
    ) -> impl Future<Output = Result<Devices, ConnectError>> + 'static + use<> {
        let devices = self.0.devices_async();
        async move { Ok(devices.await?.into_iter().map(Device::new).collect()) }
    }
}

impl RemoteHost {
    /// The server's devices for [`Device::enumerate`], which cannot return an error.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn enumerate(&self) -> Devices {
        self.devices().unwrap_or_else(|err| {
            panic!("Cannot list the devices of the remote server {self:?}: {err}")
        })
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn enumerate(&self) -> Devices {
        panic!(
            "Listing a remote server's devices blocks, which a browser cannot do: use \
             `RemoteHost::devices_async`"
        )
    }
}

/// Which device of a [`RemoteHost`] to connect. Built by [`Device::remote_options`]; nothing
/// connects until [`init`](Self::init) or [`init_async`](Self::init_async).
#[must_use = "remote options do nothing until initialized"]
#[derive(Clone, Debug)]
pub struct RemoteOptions {
    host: RemoteHost,
    device_index: DeviceIndex,
}

impl RemoteOptions {
    /// Which of the server's devices, by its position in the list the server hosts. Device 0
    /// unless set.
    pub fn device_index(mut self, index: impl Into<DeviceIndex>) -> Self {
        self.device_index = index.into();
        self
    }

    /// Open the device's session and wait for the server's answer. A device connected before is
    /// returned with the session it already has.
    ///
    /// Every device keeps its id and its runner thread for the life of the process. A process can
    /// connect 65,536 devices, and panics on the next.
    #[cfg(not(target_family = "wasm"))]
    pub fn init(self) -> Result<Device, ConnectError> {
        Ok(Device::new(
            self.host.0.connect(self.device_index.resolve())?,
        ))
    }

    /// Open the device's session from any executor. Dropping the future does not cancel a connect
    /// that has started: it finishes in the background.
    #[cfg(not(target_family = "wasm"))]
    pub fn init_async(
        self,
    ) -> impl Future<Output = Result<Device, ConnectError>> + Send + 'static + use<> {
        let device = self.host.0.connect_async(self.device_index.resolve());
        async move { Ok(Device::new(device.await?)) }
    }

    /// Open the device's session.
    #[cfg(target_family = "wasm")]
    pub fn init_async(
        self,
    ) -> impl Future<Output = Result<Device, ConnectError>> + 'static + use<> {
        let device = self.host.0.connect_async(self.device_index.resolve());
        async move { Ok(Device::new(device.await?)) }
    }
}

impl Device {
    /// Options for connecting a device on the remote server `host`.
    ///
    /// Unlike `Device::wgpu`, `init` connects before returning: a remote device fails on the
    /// network, not on its settings, so connecting reports that up front.
    pub fn remote_options(host: &RemoteHost) -> RemoteOptions {
        RemoteOptions {
            host: host.clone(),
            device_index: DeviceIndex::default(),
        }
    }
}
