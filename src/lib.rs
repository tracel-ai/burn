#[macro_use]
extern crate derive_new;

pub mod data;
pub mod module;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod train;

#[cfg(test)]
pub type TestBackend = crate::tensor::back::NdArray<f32>;
