#![warn(missing_docs)]

//! Burn JIT Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

/// Compute related module.
pub mod compute;
/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

/// Useful in Cube, should be moved over there
pub mod codegen;
pub(crate) mod tune;

mod element;
pub use codegen::compiler::{Compiler, CompilerRepresentation};
pub use codegen::dialect::gpu;

pub use element::{FloatElement, IntElement, JitElement};

mod backend;
mod bridge;
mod runtime;

pub use backend::*;
pub use bridge::*;
pub use runtime::*;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(feature = "template")]
/// Module for compiling custom non-jit kernels
pub mod template;

#[cfg(feature = "export_tests")]
pub mod tests;
