pub(crate) mod local_comm;
pub(crate) mod pump;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod spawn;
pub(crate) mod transfer;
pub(crate) mod worker;

mod authorize;
mod backend;
mod error;
mod logging;
mod settings;
#[cfg(not(target_family = "wasm"))]
mod transport;

pub use authorize::{
    AllowAll, AuthorizationRequest, ClientId, EmptyToken, PeerAuthorizer, TokenAuthorizer,
};
pub use backend::BackendServer;
pub use burn_router::{CustomOpHandler, CustomOpRegistry};
pub use error::ServeError;
pub use logging::ServerLogging;
#[doc(hidden)]
pub use settings::ServerSettings;
pub(crate) use settings::SessionSetup;
#[cfg(not(target_family = "wasm"))]
pub use transport::Transport;

#[cfg(feature = "iroh")]
pub use crate::transport::iroh::IrohIdentity;
#[cfg(all(feature = "iroh", not(target_family = "wasm")))]
pub use crate::transport::iroh::IrohTransport;
#[cfg(feature = "iroh")]
pub use crate::transport::iroh::protocol::RemoteProtocol;
#[cfg(all(feature = "websocket", not(target_family = "wasm")))]
pub use crate::transport::websocket::WebSocketTransport;
