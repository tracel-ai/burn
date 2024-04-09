mod client;
mod memory_management;
mod server;

#[cfg(not(feature = "std"))]
pub mod mutex;

#[cfg(feature = "std")]
pub mod mspc;

pub use client::*;
