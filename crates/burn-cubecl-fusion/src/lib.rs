//! Fused CubeCL kernels for `burn-fusion`.
//!
//! This crate is the CubeCL side of kernel fusion. `burn-fusion` decides which queued
//! operations can be grouped; this crate generates and launches the fused kernels for them on
//! any CubeCL runtime. [`engine`] traces a group of operations and compiles it into one
//! kernel, and [`optim`] holds the fused optimizations: element-wise chains, and matmul and
//! reductions with their surrounding element-wise work.
//!
//! It is used by `burn-cubecl` when its `fusion` feature is on (the default). Applications do
//! not use this crate directly.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `autotune` (default): benchmark fused kernel variants at runtime and keep the fastest.
//! - `autotune-checks`: check autotuned variants against each other for correctness.
//! - `tracing`: instrument operations with the `tracing` crate.

// TODO: remove when fixed in cubecl
#![allow(semicolon_in_expressions_from_non_local_macros)]

#[macro_use]
extern crate derive_new;

pub mod optim;

#[cfg(feature = "test-util")]
pub mod inspect;

mod base;

pub mod engine;
pub(crate) mod tune;

pub use base::*;
