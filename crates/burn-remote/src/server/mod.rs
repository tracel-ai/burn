pub(crate) mod local_comm;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod spawn;
#[cfg(feature = "iroh")]
pub(crate) mod time;
pub(crate) mod transfer;
pub(crate) mod worker;

mod base;
mod builder;
#[cfg(feature = "iroh")]
mod iroh;

pub use builder::{Channel, RemoteServerBuilder};
pub use burn_router::{CustomOpHandler, CustomOpRegistry};

#[cfg(feature = "iroh")]
pub use iroh::{
    AllowAll, AuthorizationRequest, IrohRemoteProtocol, PeerAuthorizer, RemoteProtocol,
};
