//! Iroh transport implementation.
//!
//! Self-contained: everything `cfg(feature = "iroh")`-specific that the session, transfer, and
//! client layers depend on lives under this module.

mod relays;
pub use relays::{InvalidRelays, IrohRelays};

#[cfg(feature = "client")]
mod host;
#[cfg(feature = "client")]
pub use host::IrohHost;

mod identity;
pub use identity::IrohIdentity;

#[cfg(all(feature = "server", not(target_family = "wasm")))]
mod server;
#[cfg(all(feature = "server", not(target_family = "wasm")))]
pub(crate) use server::IrohListener;
#[cfg(all(feature = "server", not(target_family = "wasm")))]
pub use server::IrohTransport;

mod link;
pub(crate) mod node;

#[cfg(feature = "server")]
pub(crate) mod protocol;
#[cfg(feature = "server")]
mod transfer;
#[cfg(feature = "server")]
pub(crate) use transfer::IrohTransfer;
