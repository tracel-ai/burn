use burn_router::BackendRouter;
use client::WsChannel;

#[macro_use]
extern crate derive_new;

pub mod client;
pub mod server;

pub(crate) mod shared;

pub type HttpBackend = BackendRouter<WsChannel>;
