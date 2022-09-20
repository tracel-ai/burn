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
pub type TestBackend = crate::tensor::backend::TchBackend<f32>;
#[cfg(test)]
pub type TestADBackend = crate::tensor::backend::TchADBackend<f32>;
