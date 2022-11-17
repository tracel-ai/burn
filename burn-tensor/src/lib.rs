#[macro_use]
extern crate derive_new;

pub(crate) mod graph;
pub use graph::grad::Gradients;

mod tensor;

#[cfg(feature = "export_tests")]
mod tests;
#[cfg(all(test, not(feature = "export_tests")))]
mod tests;

pub use half::f16;
pub use tensor::*;
