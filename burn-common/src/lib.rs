#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Burn Common Library
//!
//! This library contains common types used by other Burn crates that must be shared.

/// Id module contains types for unique identifiers.
pub mod id;

/// Rand module contains types for random number generation for non-std environments and for
/// std environments.
pub mod rand;

/// Stub module contains types for stubs for non-std environments and for std environments.
pub mod stub;

extern crate alloc;
