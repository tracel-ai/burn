pub(crate) mod local_comm;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod worker;

mod base;
mod builder;

pub use base::RemoteServer;
pub use builder::{Channel, RemoteServerBuilder};
pub use burn_router::{CustomOpHandler, CustomOpRegistry};
