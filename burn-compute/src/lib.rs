#![cfg(not(feature = "std"))]
#![no_std]

extern crate alloc;

mod id;

/// Compute channel module.
pub mod channel;
/// Compute client module.
pub mod client;

/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

mod compute;
pub use compute::*;
