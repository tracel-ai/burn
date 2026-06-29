//! Iroh transport implementation.
//!
//! Self-contained: everything `cfg(feature = "iroh")`-specific that the session, transfer, and
//! client layers depend on lives under this module.

mod secret;
pub use secret::RemoteSecret;

mod link;
pub mod node;

#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
mod time;
#[cfg(feature = "server")]
mod transfer;
#[cfg(feature = "server")]
pub(crate) use transfer::IrohTransfer;
