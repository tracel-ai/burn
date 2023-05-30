#[macro_use]
extern crate derive_new;

mod ops;

pub(crate) mod context;
pub(crate) mod element;
pub(crate) mod kernel;
pub(crate) mod pool;
pub(crate) mod tensor;

mod device;
pub use device::*;

mod backend;
pub use backend::*;

mod graphics;
pub use graphics::*;

#[cfg(test)]
mod tests {
    type TestBackend = crate::WGPUBackend<crate::Vulkan, f32, i64>;
    type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_add!();
    burn_tensor::testgen_sub!();

    // Once all operations will be implemented.
    // burn_tensor::testgen_all!();
}
