#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Burn
//!
//! Burn is a new comprehensive dynamic Deep Learning Framework built using Rust
//! with extreme flexibility, compute efficiency and portability as its primary goals.
//!
//! ## Performance
//!
//! Because we believe the goal of a deep learning framework is to convert computation
//! into useful intelligence, we have made performance a core pillar of Burn.
//! We strive to achieve top efficiency by leveraging multiple optimization techniques:
//!
//! - Automatic kernel fusion
//! - Asynchronous execution
//! - Thread-safe building blocks
//! - Intelligent memory management
//! - Automatic kernel selection
//! - Hardware specific features
//! - Custom Backend Extension
//!
//! ## Training & Inference
//!
//! The whole deep learning workflow is made easy with Burn, as you can monitor your training progress
//! with an ergonomic dashboard, and run inference everywhere from embedded devices to large GPU clusters.
//!
//! Burn was built from the ground up with training and inference in mind. It's also worth noting how Burn,
//! in comparison to frameworks like PyTorch, simplifies the transition from training to deployment,
//! eliminating the need for code changes.
//!
//! ## Backends
//!
//! Burn strives to be as fast as possible on as many hardwares as possible, with robust implementations.
//! We believe this flexibility is crucial for modern needs where you may train your models in the cloud,
//! then deploy on customer hardwares, which vary from user to user.
//!
//! Compared to other frameworks, Burn has a very different approach to supporting many backends.
//! By design, most code is generic over the Backend trait, which allows us to build Burn with swappable backends.
//! This makes composing backend possible, augmenting them with additional functionalities such as
//! autodifferentiation and automatic kernel fusion.
//!
//! - WGPU (WebGPU): Cross-Platform GPU Backend
//! - Candle: Backend using the Candle bindings
//! - LibTorch: Backend using the LibTorch bindings
//! - NdArray: Backend using the NdArray primitive as data structure
//! - Autodiff: Backend decorator that brings backpropagation to any backend
//! - Fusion: Backend decorator that brings kernel fusion to backends that support it
//!
//! # Quantization
//!
//! Quantization techniques perform computations and store tensors in lower precision data types like
//! 8-bit integer instead of floating point precision. There are multiple approaches to quantize a deep
//! learning model categorized as post-training quantization (PTQ) and quantization aware training (QAT).
//!
//! In post-training quantization, the model is trained in floating point precision and later converted
//! to the lower precision data type. There are two types of post-training quantization:
//!
//! 1. Static quantization: quantizes the weights and activations of the model. Quantizing the
//!    activations statically requires data to be calibrated (i.e., recording the activation values to
//!    compute the optimal quantization parameters with representative data).
//! 2. Dynamic quantization: quantized the weights ahead of time (like static quantization) but the
//!    activations are dynamically at runtime.
//!
//! Sometimes post-training quantization is not able to achieve acceptable task accuracy. In general,
//! this is where quantization-aware training (QAT) can be used: during training, fake-quantization
//! modules are inserted in the forward and backward passes to simulate quantization effects, allowing
//! the model to learn representations that are more robust to reduced precision.
//!
//! Burn does not currently support QAT. Only post-training quantization (PTQ) is implemented at this
//! time.
//!
//! Quantization support in Burn is currently in active development. It supports the following PTQ modes on some backends:
//! - Per-tensor and per-block quantization to 8-bit, 4-bit and 2-bit representations
//!
//! ## Feature Flags
//!
//! The following feature flags are available.
//! By default, the feature `std` is activated.
//!
//! - Training
//!   - `train`: Enables features `dataset` and `autodiff` and provides a training environment
//!   - `tui`: Includes Text UI with progress bar and plots
//!   - `metrics`: Includes system info metrics (CPU/GPU usage, etc.)
//! - Dataset
//!   - `dataset`: Includes a datasets library
//!   - `audio`: Enables audio datasets (SpeechCommandsDataset)
//!   - `sqlite`: Stores datasets in SQLite database
//!   - `sqlite_bundled`: Use bundled version of SQLite
//!   - `vision`: Enables vision datasets (MnistDataset)
//! - Backends
//!   - `wgpu`: Makes available the WGPU backend
//!   - `webgpu`: Makes available the `wgpu` backend with the WebGPU Shading Language (WGSL) compiler
//!   - `vulkan`: Makes available the `wgpu` backend with the alternative SPIR-V compiler
//!   - `cuda`: Makes available the CUDA backend
//!   - `rocm`: Makes available the ROCm backend
//!   - `candle`: Makes available the Candle backend
//!   - `tch`: Makes available the LibTorch backend
//!   - `ndarray`: Makes available the NdArray backend
//! - Backend specifications
//!   - `accelerate`: If supported, Accelerate will be used
//!   - `blas-netlib`: If supported, Blas Netlib will be use
//!   - `openblas`: If supported, Openblas will be use
//!   - `openblas-system`: If supported, Openblas installed on the system will be use
//!   - `autotune`: Enable running benchmarks to select the best kernel in backends that support it.
//!   - `fusion`: Enable operation fusion in backends that support it.
//! - Backend decorators
//!   - `autodiff`: Makes available the Autodiff backend
//! - Model Storage
//!   - `store`: Enables model storage with SafeTensors format and PyTorch interoperability
//! - Others:
//!   - `std`: Activates the standard library (deactivate for no_std)
//!   - `server`: Enables the remote server.
//!   - `network`: Enables network utilities (currently, only a file downloader with progress bar)
//!   - `experimental-named-tensor`: Enables named tensors (experimental)
//!
//! You can also check the details in sub-crates [`burn-core`](https://docs.rs/burn-core) and [`burn-train`](https://docs.rs/burn-train).

pub use burn_core::*;

/// Train module
#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}

/// Backend module.
pub mod backend;

#[cfg(feature = "server")]
pub use burn_remote::server;

/// Module for collective operations
#[cfg(feature = "collective")]
pub mod collective;

/// Module for model storage and serialization
#[cfg(feature = "store")]
pub mod store {
    pub use burn_store::*;
}

/// Neural network module.
pub mod nn {
    pub use burn_nn::*;
}

/// Optimizers module.
pub mod optim {
    pub use burn_optim::*;
}

// For backward compat, `burn::lr_scheduler::*`
/// Learning rate scheduler module.
#[cfg(feature = "std")]
pub mod lr_scheduler {
    pub use burn_optim::lr_scheduler::*;
}
// For backward compat, `burn::grad_clipping::*`
/// Gradient clipping module.
pub mod grad_clipping {
    pub use burn_optim::grad_clipping::*;
}

/// CubeCL module re-export.
#[cfg(feature = "cubecl")]
pub mod cubecl {
    pub use cubecl::*;
}

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! burn::prelude::*` to your code to quickly get started with
    //! Burn.
    pub use burn_core::prelude::*;

    pub use crate::nn;
}
