#![warn(missing_docs)]

//! Burn JIT Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

// Compute related module.
pub mod compute {
    pub use burn_cube::*;
}
// pub mod compute;

/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

/// Useful in Cube, should be moved over there
// pub mod codegen;
pub mod codegen {
    pub mod dialect {
        pub use burn_cube::codegen::dialect as gpu;
        pub use burn_cube::codegen::dialect::*;
    }
    pub use burn_cube::cpa as gpu;
    pub use burn_cube::*;
}

pub(crate) mod tune;

pub mod element;
pub use burn_cube::codegen::dialect as gpu;
pub use burn_cube::{Compiler, CompilerRepresentation};

pub use element::{FloatElement, IntElement, JitElement};

mod backend;
mod bridge;

pub use backend::*;
pub use bridge::*;
pub use burn_cube::*;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(feature = "template")]
/// Module for compiling custom non-jit kernels
pub mod template;

#[cfg(feature = "export_tests")]
pub mod tests;
