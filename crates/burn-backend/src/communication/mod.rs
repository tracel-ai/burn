//! Types and helpers for inter-device operations.

pub(crate) mod all_reduce;
#[cfg(feature = "communication")]
pub(crate) mod api;
mod base;
#[cfg(feature = "communication")]
pub(crate) mod client;
#[cfg(feature = "communication")]
pub(crate) mod server;

#[cfg(feature = "communication")]
pub use api::*;
pub use base::*;
