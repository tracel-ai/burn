#![feature(generic_associated_types)]

#[macro_use]
extern crate derive_new;

pub(crate) mod graph;

mod tensor;
pub use tensor::*;
