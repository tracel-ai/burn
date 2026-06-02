#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Standard Library
//!
//! This library contains core types and utilities shared across Burn, including shapes, indexing,
//! and data types.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Id module contains types for unique identifiers.
pub mod id;

/// Tensor utilities.
pub mod tensor;
pub use tensor::*;

/// Tensor data representation and helpers.
pub mod data;
pub use data::*;

/// Random value distributions.
pub mod distribution;
pub use distribution::*;

/// Traits for tensor element types and conversions.
pub mod element;
pub use element::*;

mod device_settings;
pub use device_settings::*;

/// Distributed configurations.
pub mod distributed;

/// Configuration types for tensor operations (conv, pool, interpolate, pad, etc).
pub mod ops;
pub use ops::*;

/// Burn runtime configurations.
pub mod config;

/// Common Errors.
pub use cubecl_zspace::errors::{self, *};

/// Network utilities.
#[cfg(feature = "network")]
pub mod network;

/// An ID unique to any unordered combination of devices, used by collective /
/// communication primitives (distributed training etc.).
///
/// Mirrors `cubecl_runtime::server::CommunicationId` so that the
/// `burn_fusion::FusionUtilities::initialized_comms` set (and other consumers)
/// can be reused without depending on cubecl directly.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct CommunicationId {
    /// Stable hash of the (sorted) set of device ids that participate.
    pub id: u64,
}

impl From<alloc::vec::Vec<cubecl_common::device::DeviceId>> for CommunicationId {
    fn from(mut value: alloc::vec::Vec<cubecl_common::device::DeviceId>) -> Self {
        use core::hash::{Hash, Hasher};
        // Sort so any permutation of the same devices yields the same id.
        value.sort();
        let mut hasher = ahash::AHasher::default();
        value.hash(&mut hasher);
        CommunicationId {
            id: hasher.finish(),
        }
    }
}

pub use cubecl_common::bytes::*;
pub use cubecl_common::device_handle::DeviceHandle;
pub use cubecl_common::*;
pub use half::{bf16, f16};

pub use cubecl_common::flex32;
