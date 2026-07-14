//! Vision ops for burn, with GPU acceleration where possible.
//!
//! # Operations
//! Operation names are based on `opencv` wherever applicable.
//!
//! Currently implemented are:
//! - `connected_components`
//! - `connected_components_with_stats`
//! - `nms` (Non-Maximum Suppression)
//!

#![warn(missing_docs)]

extern crate alloc;

macro_rules! cfg_backend {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "cubecl-backend", feature = "tch", feature = "flex"))]
            $item
        )*
    }
}

cfg_backend! {
    /// Backend implementations for JIT and CPU
    pub mod backends;
    mod base;
    mod ops;
    mod tensor;
    pub use base::*;
    pub use ops::*;
    pub use tensor::*;
    pub use backends::{KernelShape, create_structuring_element};
}

mod transform;
pub use transform::*;

/// Module for vision/image utilities
pub mod utils;

#[cfg(feature = "loss")]
#[cfg_attr(docsrs, doc(cfg(feature = "loss")))]
/// Loss module
pub mod loss;

/// Dispatches connected components based on the int dtype, binding a concrete
/// integer type to enable generic instantiations without extra trait bounds (after removing
/// `ElementComparison` from `Element`).
#[macro_export]
macro_rules! dispatch_int_dtype {
    ($dtype:expr, |$ty:ident| $body:expr) => {
        match $dtype {
            burn_core::tensor::IntDType::I64 => {
                type $ty = i64;
                $body
            }
            burn_core::tensor::IntDType::I32 => {
                type $ty = i32;
                $body
            }
            burn_core::tensor::IntDType::I16 => {
                type $ty = i16;
                $body
            }
            burn_core::tensor::IntDType::I8 => {
                type $ty = i8;
                $body
            }
            burn_core::tensor::IntDType::U64 => {
                type $ty = u64;
                $body
            }
            burn_core::tensor::IntDType::U32 => {
                type $ty = u32;
                $body
            }
            burn_core::tensor::IntDType::U16 => {
                type $ty = u16;
                $body
            }
            burn_core::tensor::IntDType::U8 => {
                type $ty = u8;
                $body
            }
        }
    };
}

/// Dispatches connected components based on the bool dtype, binding a concrete
/// integer type to enable generic instantiations without extra trait bounds (after removing
/// `ElementComparison` from `Element`).
#[macro_export]
macro_rules! dispatch_bool_dtype {
    ($dtype:expr, |$ty:ident| $body:expr) => {
        match $dtype {
            burn_core::tensor::BoolStore::Native => {
                type $ty = bool;
                $body
            }
            burn_core::tensor::BoolStore::U8 => {
                type $ty = u8;
                $body
            }
            burn_core::tensor::BoolStore::U32 => {
                type $ty = u32;
                $body
            }
        }
    };
}
