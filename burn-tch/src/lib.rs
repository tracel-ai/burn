mod backend;
mod element;
mod module_ops;
mod ops;
mod tensor;
mod tensor_ops;

pub use backend::*;
pub use tensor::*;
pub use tensor_ops::*;

#[cfg(test)]
mod tests {
    type TestBackend = crate::TchBackend<f32>;

    burn_tensor::test_all!();
}
