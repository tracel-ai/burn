//! Client/server networking used by Burn's remote backend.
//!
//! [`Protocol`] abstracts a transport with a [`ProtocolServer`] that routes connections to
//! handlers and a [`ProtocolClient`] that opens [`CommunicationChannel`]s to an [`Address`].
//!
//! - `websocket` (feature `websocket`): a WebSocket implementation of [`Protocol`].
//! - `external_comm` (feature `data-service`): lets one server download a tensor directly from
//!   another, without routing the data through the client.
//!
//! This crate is an implementation detail of `burn-remote`; applications do not use it directly.
#[macro_use]
extern crate derive_new;

mod base;
pub use base::*;

pub mod util;

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(feature = "data-service")]
pub mod external_comm;
