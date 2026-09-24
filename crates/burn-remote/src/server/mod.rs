pub(crate) mod local_comm;
pub(crate) mod pump;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod transfer;
pub(crate) mod worker;

mod builder;
mod logging;

pub use builder::{Channel, RemoteServerBuilder};
pub use burn_router::{CustomOpHandler, CustomOpRegistry};
pub use logging::ServerLogging;

#[cfg(feature = "iroh")]
pub use crate::transport::iroh::protocol::{
    AllowAll, AuthorizationRequest, IrohRemoteProtocol, PeerAuthorizer, RemoteProtocol,
};
