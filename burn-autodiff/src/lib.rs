#[macro_use]
extern crate derive_new;

pub(crate) mod graph;
pub(crate) mod ops;
pub(crate) mod tensor;

mod backend;
pub use backend::*;
