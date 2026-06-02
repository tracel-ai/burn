#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

//! Burn multi-backend router.

mod backend;
mod bridge;
mod channel;
mod client;
mod interpreter;
mod ops;
mod tensor;

pub use backend::*;
pub use bridge::*;
pub use channel::*;
pub use client::*;
pub use interpreter::*;
pub use tensor::*;

extern crate alloc;
