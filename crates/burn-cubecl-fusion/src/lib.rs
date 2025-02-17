#[macro_use]
extern crate derive_new;

pub mod elemwise;
pub mod matmul;

mod base;

pub(crate) mod on_write;
pub(crate) mod tune;

pub use base::*;
