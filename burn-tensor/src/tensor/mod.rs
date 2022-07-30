pub mod backend;
pub mod ops;

mod data;
mod print;
mod shape;
mod tensor;
mod tensor_trait;

pub use data::*;
pub use shape::*;
pub use tensor::*;
pub use tensor_trait::*;
