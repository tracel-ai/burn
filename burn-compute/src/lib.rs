#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! Burn compute crate that helps creating high performance async backends.

extern crate alloc;

#[macro_use]
extern crate derive_new;

mod id;

/// Compute channel module.
pub mod channel;
/// Compute client module.
pub mod client;

/// Autotune module, only available with std for now
#[cfg(feature = "std")]
pub mod tune;

pub(crate) mod autotune_server;
/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

mod compute;
pub use compute::*;
