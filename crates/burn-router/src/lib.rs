#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

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

/// A local channel with a simple byte bridge between backends.
/// It transfers tensors between backends via the underlying [tensor data](burn_backend::TensorData).
pub type DirectByteChannel<Backends> = DirectChannel<Backends, ByteBridge<Backends>>;

/// Router backend.
///
/// # Example
///
/// ```ignore
/// type MyBackend = Router<(NdArray, Wgpu)>;
/// ```
pub type Router<Backends> = BackendRouter<DirectByteChannel<Backends>>;

extern crate alloc;

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::BackendRouter;
    use crate::DirectByteChannel;

    pub type TestBackend1 = burn_ndarray::NdArray<f32, i32>;
    pub type TestBackend2 = burn_wgpu::Wgpu<f32, i32>;
    pub type TestBackend = BackendRouter<DirectByteChannel<(TestBackend1, TestBackend2)>>;
}
