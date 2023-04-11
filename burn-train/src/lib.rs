#[macro_use]
extern crate derive_new;

pub mod checkpoint;
pub mod logger;
pub mod metric;

mod callback;
mod learner;

pub use callback::*;
pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArrayBackend<f32>;
