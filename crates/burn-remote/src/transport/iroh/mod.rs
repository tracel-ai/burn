//! Iroh transport implementation.
//!
//! Self-contained: everything `cfg(feature = "iroh")`-specific that the session, transfer, and
//! client layers depend on lives under this module.

mod relays;
mod secret;
pub use relays::IrohRelays;
pub use secret::RemoteSecret;

#[cfg(feature = "client")]
mod host;
#[cfg(feature = "client")]
pub use host::IrohHost;

#[cfg(feature = "server")]
mod channel;
#[cfg(feature = "server")]
pub use channel::{IrohChannel, IrohChannelBuilder};

mod link;
pub mod node;

#[cfg(feature = "server")]
pub mod protocol;
#[cfg(feature = "server")]
mod transfer;
#[cfg(feature = "server")]
pub(crate) use transfer::IrohTransfer;
