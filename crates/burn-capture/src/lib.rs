#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Non-executing backend for capturing Burn operation graphs.

extern crate alloc;

mod capture;

pub use burn_ir::TensorId;
pub use capture::*;
