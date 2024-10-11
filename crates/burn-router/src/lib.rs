#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn multi-backend router.

mod backend;
mod bridge;
mod channel;
mod client;
mod ops;
mod runner;
mod tensor;

pub use backend::*;
pub use bridge::*;
pub use channel::*;
pub use client::*;
pub use runner::*;
pub use tensor::*;

extern crate alloc;
