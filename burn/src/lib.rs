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
//! - Backends
//!   - `wgpu`: Makes available the WGPU backend
//!   - `candle`: Makes available the Candle backend
//!   - `tch`: Makes available the LibTorch backend
//!   - `ndarray`: Makes available the NdArray backend
//! - Backend specifications
//!   - `cuda`: If supported, CUDA will be used
//!   - `accelerate`: If supported, Accelerate will be used
//!   - `blas-netlib`: If supported, Blas Netlib will be use
//!   - `openblas`: If supported, Openblas will be use
//!   - `openblas-system`: If supported, Openblas installed on the system will be use
//!   - `wasm-sync`: When targeting wasm, but want a sync API (won't work with WGPU)
//! - Backend decorators
//!   - `autodiff`: Makes available the Autodiff backend
//!   - `fusion`: Makes available the Fusion backend
//! - Others:
//!   - `std`: Activates the standard library (deactivate for no_std)
//!   - `experimental-named-tensor`: Enables named tensors (experimental)

pub use burn_core::*;

/// Train module
#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}
