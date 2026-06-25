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
pub use ::iroh::protocol::{Router, RouterBuilder};
#[cfg(feature = "iroh")]
pub use iroh::{AuthorizationRequest, IrohRemoteProtocol, PeerAuthorizer, RemoteProtocol};
// The blocking process entry points exist only on native targets; the browser server is driven by
// the JS event loop and composed through `RemoteNode::protocol(...).serve()` directly.
#[cfg(all(feature = "iroh", not(target_family = "wasm")))]
pub use iroh::{start_iroh, start_iroh_async};
