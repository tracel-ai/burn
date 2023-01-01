#[macro_use]
extern crate derive_new;

pub mod config;
pub mod data;
pub mod module;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod train;

#[cfg(test)]
pub type TestBackend = burn_ndarray::NdArrayBackend<f32>;
#[cfg(test)]
pub type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;
