#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn
//!
//! Burn is a deep learning framework written in Rust. It covers tensors, automatic
//! differentiation, neural network modules, optimizers, training and model storage, and runs
//! the same model code on GPUs (CUDA, ROCm, Metal, Vulkan, WebGPU), CPUs and WebAssembly.
//!
//! ## Quick start
//!
//! Burn ships no execution backend by default. Enable one or more with Cargo features:
//!
//! ```toml
//! [dependencies]
//! burn = { version = "0.22", features = ["wgpu"] }
//! ```
//!
//! Models are plain structs that derive [`Module`](module::Module). Tensors carry their rank in
//! the type and their backend in their [`Device`](tensor::Device), so model code has no backend
//! type parameter:
//!
//! ```rust,no_run
//! use burn::nn::{Linear, LinearConfig, Relu};
//! use burn::prelude::*;
//!
//! #[derive(Module, Debug)]
//! struct Mlp {
//!     hidden: Linear,
//!     activation: Relu,
//!     output: Linear,
//! }
//!
//! impl Mlp {
//!     fn new(device: &Device) -> Self {
//!         Self {
//!             hidden: LinearConfig::new(784, 128).init(device),
//!             activation: Relu::new(),
//!             output: LinearConfig::new(128, 10).init(device),
//!         }
//!     }
//!
//!     fn forward(&self, input: Tensor<2>) -> Tensor<2> {
//!         let x = self.activation.forward(self.hidden.forward(input));
//!         self.output.forward(x)
//!     }
//! }
//!
//! // An enabled backend in priority order (GPUs before CPUs), unless `BURN_DEVICE` names one.
//! // `Device::wgpu(..)`, `Device::cuda(0)`, ... pick one explicitly.
//! let device = Device::default();
//! let model = Mlp::new(&device);
//! let logits = model.forward(Tensor::zeros([32, 784], &device));
//! ```
//!
//! The [Burn Book](https://burn.dev/books/burn/) walks through a full training workflow.
//!
//! ## Crate map
//!
//! - [`tensor`]: [`Tensor`], [`Device`](tensor::Device), dtypes and tensor
//!   operations.
//! - [`module`] and [`nn`]: the [`Module`](module::Module) trait and neural network layers.
//! - [`config`]: serializable configuration structs with `#[derive(Config)]`.
//! - [`optim`], [`lr_scheduler`], [`grad_clipping`]: optimizers and training utilities.
//! - [`data`]: datasets, transformations and data loaders.
//! - [`store`]: saving and loading weights in burnpack, SafeTensors and PyTorch formats.
//! - `train`: the `Learner`, metrics and the training dashboard (`train` feature).
//! - `vision`, `signal`, `linalg`: domain-specific tensor operations (features of the same
//!   names).
//! - `remote` and `server`: run tensors on devices hosted by another machine (`remote` and
//!   `remote-server` features).
//! - [`prelude`]: the types most programs import.
//!
//! ## Backends
//!
//! Every enabled backend is available at runtime through a `Device` constructor, and several can
//! be used side by side:
//!
//! | Backend                 | Feature  | Device                               |
//! | ----------------------- | -------- | ------------------------------------ |
//! | CUDA                    | `cuda`   | `Device::cuda(0)`                    |
//! | ROCm                    | `rocm`   | `Device::rocm(0)`                    |
//! | wgpu (any graphics API) | `wgpu`   | `Device::wgpu(Default::default())`   |
//! | Metal                   | `metal`  | `Device::metal(Default::default())`  |
//! | Vulkan                  | `vulkan` | `Device::vulkan(Default::default())` |
//! | WebGPU                  | `webgpu` | `Device::webgpu(Default::default())` |
//! | CubeCL CPU              | `cpu`    | `Device::cpu()`                      |
//! | Flex (pure Rust CPU)    | `flex`   | `Device::flex()`                     |
//!
//! Autodiff and kernel fusion are decorators over these backends: `device.autodiff()` enables
//! gradients for tensors created on a device, and the CubeCL backends fuse operations by default.
//!
//! ## Quantization
//!
//! Burn supports post-training quantization of weights and activations, per tensor or per
//! block, to 8, 4 and 2-bit integers and to FP8 and FP4 formats on supported backends.
//! Quantization-aware training is not supported yet. See the
//! [quantization chapter](https://burn.dev/books/burn/performance/quantization.html).
//!
//! ## Feature Flags
//!
//! The following feature flags are available.
//! Default features include `std` and `optim` (and therefore `autodiff`), but no execution backend.
//! Select a backend explicitly, for example `features = ["wgpu"]` or `["flex"]`.
//! Specialized operations are also opt-in, for example `features = ["flex", "signal"]`.
//! Backend-free builds can define tensor/model APIs without installing an execution backend.
//! `Device::default()` panics if no execution backend is available; graph capture remains
//! available through `Device::capture()` with the `capture` feature.
//!
//! - Training
//!   - `train`: Enables features `dataset` and `optim` and provides a training environment
//!   - `optim`: Enables optimizers and learning rate schedulers (implies `autodiff`)
//!   - `rl`: Enables reinforcement learning utilities
//!   - `tui`: Includes Text UI with progress bar and plots (requires `train`)
//!   - `metrics`: Includes system info metrics (CPU/GPU usage, etc.) (requires `train`)
//! - Dataset
//!   - `dataset`: Includes a datasets library
//!   - `audio`: Enables audio datasets (SpeechCommandsDataset)
//!   - `sqlite`: Stores datasets in an SQLite database, backed by [Turso](https://turso.tech/)
//!   - `sqlite-bundled`: Deprecated alias for `sqlite`
//!   - `vision`: Enables vision datasets (MnistDataset) and the `burn-vision` ops module
//! - Backends
//!   - `wgpu`: Makes available the WGPU backend, on whichever graphics API the platform provides
//!   - `webgpu`: Adds `Device::webgpu`, pinned to WebGPU (implies `wgpu`)
//!   - `vulkan`: Adds `Device::vulkan`, pinned to Vulkan (implies `wgpu`)
//!   - `metal`: Adds `Device::metal`, pinned to Metal with native MSL (implies `wgpu`)
//!   - `cuda`: Makes available the CUDA backend
//!   - `rocm`: Makes available the ROCm backend
//!   - `cpu`: Makes available the CubeCL CPU backend
//!   - `flex`: Makes available the Flex backend (pure-Rust CPU, std/no_std/WASM)
//! - Backend specifications
//!   - `simd`: Enable SIMD kernels in the Flex backend
//!   - `rayon`: Enable multi-threaded execution in the Flex backend
//!   - `autotune`: Enable running benchmarks to select the best kernel in backends that support it.
//!   - `autotune-checks`: Check that every autotune candidate produces the same output (debugging).
//!   - `persistence`: Enable persistent CubeCL caches across process runs, including when default
//!     features are disabled. Compiled-kernel caching also requires `compilation.cache = true`
//!     in the CubeCL runtime configuration. Does not select a backend.
//!   - `x86-v4`: Enable AVX-512 matmul kernels in the Flex backend.
//!   - `apple-amx`: Enable the experimental Apple AMX matmul kernels in the Flex backend.
//!   - `template`: Enable hand-written, non-JIT custom kernels in the CubeCL backends.
//!   - `fusion`: Enable operation fusion in backends that support it.
//!   - `tracing`: Enable diagnostic tracing in the selected backends (disabled by default).
//! - Backend decorators
//!   - `autodiff`: Makes available the Autodiff backend
//! - Model Storage
//!   - `store`: Enables the `burn-store` snapshot tooling and burnpack stores; with `std`, this
//!     also includes SafeTensors
//!   - `safetensors`: Enables SafeTensors import and export in `no_std` builds (implies `store`)
//!   - `pytorch`: Enables PyTorch checkpoint import (implies `store`)
//! - Others:
//!   - `std`: Activates the standard library (deactivate for no_std)
//!   - `linalg`: Enables linear algebra operations
//!   - `capture`: Makes the non-executing graph capture backend available.
//!   - `group`: Makes `Device::group` available, which splits each tensor over several devices
//!     for tensor parallelism.
//!   - `ir`: Makes Burn's operation intermediate representation available.
//!   - `cubecl`: Re-exports CubeCL as `burn::cubecl` for writing custom kernels.
//!   - `signal`: Enables signal processing operations from `burn-signal`.
//!   - `extension`: Enables the backend extension API, including `Tensor::from_primitive`.
//!   - `remote`: Enables remote devices over Iroh; `remote-websocket` adds the WebSocket transport.
//!   - `remote-server`: Enables the remote server (implies `remote`).
//!   - `network`: Enables network utilities (currently, only a file downloader with progress bar)
//!
//! You can also check the details in sub-crates [`burn-core`](https://docs.rs/burn-core) and [`burn-train`](https://docs.rs/burn-train).
//!
//! ### Backend tracing
//!
//! Add `"tracing"` to the features of your `burn` dependency to compile backend instrumentation,
//! including autodiff and fusion spans. When depending directly on `burn-autodiff` or
//! `burn-fusion`, enable their `tracing` feature instead. These spans are opt-in: configuring a
//! tracing subscriber alone does not enable them. Configure your subscriber to include the
//! `trace` level to observe tensor operation spans.
//!
//! The feature propagates to enabled backends without selecting an additional backend. Normal
//! training logs remain available without this feature.

pub use burn_core::*;

/// Linear algebra operations.
#[cfg(feature = "linalg")]
pub mod linalg {
    pub use burn_linalg::*;
}

/// Core module infrastructure and neural-network initializers.
pub mod module {
    pub use burn_core::module::*;
    pub use burn_nn::Initializer;
}

/// Tensor types and compatibility re-exports.
pub mod tensor {
    pub use burn_core::tensor::*;

    /// Compatibility path for signal processing operations.
    #[cfg(feature = "signal")]
    pub mod signal {
        pub use burn_signal::*;
    }

    /// Compatibility path for linear algebra operations.
    #[cfg(feature = "linalg")]
    pub mod linalg {
        pub use burn_linalg::*;
    }
}

/// Train module
#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}

/// Module for reinforcement learning.
#[cfg(feature = "rl")]
pub mod rl {
    pub use burn_rl::*;
}

#[cfg(feature = "remote")]
pub use burn_core::tensor::remote;
#[cfg(feature = "remote-server")]
pub use burn_core::tensor::server;

/// Model storage and serialization: the non-generic record system (always available), plus,
/// with the `store` feature, the snapshot tooling and burnpack stores. The `safetensors` and
/// `pytorch` features add those importers.
pub mod store {
    pub use burn_core::store::*;
    #[cfg(feature = "store")]
    pub use burn_store::*;
}

/// Neural network module.
pub mod nn {
    pub use burn_nn::*;
}

pub use burn_std::config::{BurnConfig, config as runtime_config};

#[cfg(all(test, feature = "capture"))]
mod capture_tests {
    use crate::{module::Module, nn::BatchNormConfig, tensor::Device};

    #[test]
    fn capture_feature_exposes_the_user_facing_device_api() {
        let device = Device::capture();
        let captured = device
            .capture_scope(|scope| scope.complete([], []))
            .unwrap();

        assert!(captured.graph.operations.is_empty());
    }

    #[test]
    fn shared_running_state_moves_across_capture_scopes() {
        let module = BatchNormConfig::new(3).init(&Device::default());
        let first_device = Device::capture();
        let second_device = Device::capture();

        let first = first_device
            .capture_scope(|scope| {
                let _module = module.clone().to_device(&first_device);
                scope.complete([], [])
            })
            .unwrap();
        let second = second_device
            .capture_scope(|scope| {
                let _module = module.clone().to_device(&second_device);
                scope.complete([], [])
            })
            .unwrap();

        assert_eq!(first.values.len(), 4);
        assert_eq!(second.values.len(), 4);
    }
}

/// Optimizers module.
#[cfg(feature = "optim")]
pub mod optim {
    pub use burn_optim::*;
}

// For backward compat, `burn::lr_scheduler::*`
/// Learning rate scheduler module.
#[cfg(all(feature = "optim", feature = "std"))]
pub mod lr_scheduler {
    pub use burn_optim::lr_scheduler::*;
}
// For backward compat, `burn::grad_clipping::*`
/// Gradient clipping module.
#[cfg(feature = "optim")]
pub mod grad_clipping {
    pub use burn_optim::grad_clipping::*;
}

/// CubeCL module re-export.
#[cfg(feature = "cubecl")]
pub mod cubecl {
    pub use cubecl::*;
}

#[cfg(feature = "vision")]
/// Vision module.
pub mod vision {
    pub use burn_vision::*;
}

#[cfg(feature = "signal")]
/// Signal processing module.
pub mod signal {
    pub use burn_signal::*;
}

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! burn::prelude::*` to your code to quickly get started with
    //! Burn.
    pub use burn_core::prelude::*;

    pub use crate::nn;
}
