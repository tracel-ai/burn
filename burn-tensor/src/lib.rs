#[macro_use]
extern crate derive_new;

pub(crate) mod graph;
pub use graph::grad::Gradients;

mod tensor;

pub use half::f16;
pub use tensor::*;
