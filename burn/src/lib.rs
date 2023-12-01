#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Burn
//! Burn is a new comprehensive dynamic Deep Learning Framework built using Rust
//! with extreme flexibility, compute efficiency and portability as its primary goals.
//!
//! ## Performance
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
//!
//! ## Feature Flags
//!
//! The following feature flags are available.
//!
//! - Enabled by default
//!   - `std`: Activates the standard library (deactivate for no_std)
//!   - `dataset`: Includes datasets
//!   - `sqlite`: Stores datasets in SQLite database
//! - Training
//!   - `train`: For training on datasets
//!   - `tui`: Includes Text UI with progress bar and plots
//!   - `metrics`: Includes system info metrics (CPU/GPU usage, etc.)
//! - Dataset
//!   - `audio`: enables audio dataset (SpeechCommandsDataset)
//!   - `sqlite_bundled`: Use bundled version of SQLite
//! - Backends
//!   - `wgpu`: Use the WGPU backend
//!   - `candle`: Use the Candle backend
//!   - `tch`: Use the LibTorch backend
//!   - `ndarray`: Use the NdArray backend
//! - Backend specifications
//!   - `cuda`: If backend and platform allow it, will run on CUDA GPU
//!   - `accelerate`: If backend and platform allow it, will run with Accelerate
//!   - `blas-netlib`: Netlib (NdArray only)
//!   - `openblas`: OpenBLAS static linked (NdArray only)
//!   - `openblas-system`: OpenBLAS from the system (NdArray only)
//!   - `wasm-sync`: When targetting wasm (except with WGPU)
//! - Backend decorators
//!   - `autodiff`: Activates autodifferentiation
//!   - `fusion`: Activates kernel fusion
//! - Others:
//!   - `experimental-named-tensor`: Enables named tensors (experimental)

pub use burn_core::*;

/// Train module
#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}
