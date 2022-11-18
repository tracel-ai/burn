#[macro_use]
extern crate derive_new;

mod backend;
mod element;
mod module_ops;
mod ops;
mod shape;
mod tensor;
mod tensor_ops;

pub use backend::*;
pub(crate) use tensor::*;

#[cfg(test)]
mod tests {
    type TestBackend = crate::NdArrayBackend<f32>;

    burn_tensor::testgen_all!();
}
