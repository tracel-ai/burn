#[macro_use]
extern crate derive_new;

#[cfg(feature = "client")]
pub(crate) mod client;

#[cfg(feature = "server")]
pub mod server;

pub(crate) mod shared;

#[cfg(feature = "client")]
mod __client {
    use super::*;

    use burn_router::BackendRouter;
    use client::WsChannel;

    /// The remote backend allows you to run computation on a remote device.
    ///
    /// Make sure there is a running server before trying to connect to it.
    ///
    /// ```rust, ignore
    /// fn main() {
    ///     let device = Default::default();
    ///     let port = 3000;
    ///
    ///     // You need to activate the `server` feature flag to have access to this function.
    ///     burn::server::start::<burn::backend::Wgpu>(device, port);
    /// }
    ///```
    pub type RemoteBackend = BackendRouter<WsChannel>;

    pub use client::WsDevice as RemoteDevice;
}
#[cfg(feature = "client")]
pub use __client::*;
