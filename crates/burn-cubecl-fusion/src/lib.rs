#[macro_use]
extern crate derive_new;

pub mod elemwise;
pub mod matmul;
pub mod reduce;

mod base;

pub(crate) mod shared;
pub(crate) mod tune;

pub use base::*;
