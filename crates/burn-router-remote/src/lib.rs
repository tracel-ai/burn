use burn_router::BackendRouter;
use client::WsChannel;

#[macro_use]
extern crate derive_new;

#[cfg(feature = "client")]
pub(crate) mod client;
#[cfg(feature = "server")]
pub(crate) mod server;
pub(crate) mod shared;

#[cfg(feature = "client")]
/// The remote backend allows you to run computation on a remote device.
///
/// Make sure there is a running server before trying to connect to it.
///
/// ```rust, ignore
/// fn main() {
///     let device = Default::default();
///     let port = 3000;
///
///     // You need to activate the `remote-server` feature flag to have access to this function.
///     burn::backend::remote::start_server::<burn::backend::Wgpu>(device, port);
/// }
///```
pub type RemoteBackend = BackendRouter<WsChannel>;
pub use client::WsDevice as RemoteDevice;

#[cfg(feature = "server")]
pub use server::start as start_server;
