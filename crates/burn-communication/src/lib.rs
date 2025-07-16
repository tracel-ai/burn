#[macro_use]
extern crate derive_new;

mod base;
pub use base::*;

pub mod util;
pub mod websocket;

pub use websocket::base::WsNetwork;

#[cfg(feature = "data-service")]
pub mod data_service;
