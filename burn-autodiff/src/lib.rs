#[macro_use]
extern crate derive_new;

pub(crate) mod grads;
pub(crate) mod graph;
pub(crate) mod ops;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;
pub use backend::*;

#[cfg(feature = "export_tests")]
mod tests;
