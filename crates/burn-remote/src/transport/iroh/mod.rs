//! Iroh transport implementation.
//!
//! Self-contained: everything `cfg(feature = "iroh")`-specific that the session, transfer, and
//! client layers depend on lives under this module.

mod secret;
pub use secret::RemoteSecret;

pub mod node;
mod link;

#[cfg(feature = "server")]
mod transfer;
#[cfg(feature = "server")]
pub(crate) use transfer::IrohTransfer;
