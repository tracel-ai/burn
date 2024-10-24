#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn multi-backend router.

mod backend;
mod bridge;
mod channel;
mod client;
mod ops;
mod runner;
mod tensor;
mod types;

pub use backend::*;
pub use bridge::*;
pub use channel::*;
pub use client::*;
pub use runner::*;
pub use tensor::*;
pub use types::*;

extern crate alloc;

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec;

    use crate::BackendRouter;
    use crate::ByteBridge;
    use crate::DirectChannel;

    type DirectByteChannel<Backends> = DirectChannel<Backends, ByteBridge<Backends>>;

    pub type TestBackend1 = burn_ndarray::NdArray<f32, i32>;
    pub type TestBackend2 = burn_wgpu::Wgpu<f32, i32>;
    pub type TestBackend = BackendRouter<DirectByteChannel<(TestBackend1, TestBackend2)>>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    burn_tensor::testgen_all!();
    // TODO: add support for quantization
    // burn_tensor::testgen_quantization!();

    #[cfg(feature = "std")]
    burn_autodiff::testgen_all!();
}
