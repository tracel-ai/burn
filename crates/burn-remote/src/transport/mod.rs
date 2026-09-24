//! Transport layer for Burn Remote.
//!
//! `burn-remote` owns the transport seam: identity (`PeerId`/`PeerAddr`) is transport-agnostic and
//! lives here, and each concrete transport (iroh, websocket) is a self-contained submodule that
//! plugs into the session and transfer layers. The rest of the crate is written against these
//! abstractions and stays `cfg`-free; the `cfg(feature = ...)` selection between transports is
//! contained to this module.

mod identity;
pub use identity::{PeerAddr, PeerId};

pub(crate) mod link;

#[cfg(any(feature = "iroh", all(feature = "client", feature = "websocket")))]
mod open;
#[cfg(any(feature = "iroh", all(feature = "client", feature = "websocket")))]
pub(crate) use open::OpenError;

#[cfg(feature = "iroh")]
pub mod iroh;

#[cfg(feature = "websocket")]
pub(crate) mod websocket;
