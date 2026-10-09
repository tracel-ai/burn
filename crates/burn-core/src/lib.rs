#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! The core of Burn: modules, configuration, records and data loading, on top of the tensor API.
//!
//! Applications should depend on [`burn`](https://docs.rs/burn), which re-exports everything here
//! together with neural network layers, optimizers and training. This crate is the dependency for
//! libraries that only need the core abstractions.
//!
//! - [`tensor`]: Burn's tensor API, re-exported from `burn-tensor`.
//! - [`module`]: the [`Module`](module::Module) trait, implemented with `#[derive(Module)]`, and
//!   [`Param`](module::Param) for trainable tensors.
//! - [`config`]: serializable configuration structs with `#[derive(Config)]`.
//! - [`store`]: module records and the burnpack format.
//! - [`data`]: datasets, batchers and data loaders (`std` only).
//! - [`prelude`]: the types most programs import.
//!
//! # Feature flags
//!
//! Backend features (`wgpu`, `cuda`, `flex`, ...) and the `autodiff`, `fusion`, `remote` and
//! `capture` features match those of `burn`. Others:
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `dataset`: the dataset library in [`data`]; `vision`, `audio` and `sqlite` add sources.
//! - `network`: file downloads with a progress bar.
//! - `tracing`: instrument operations with the `tracing` crate.

// `derive_new` provides the `#[derive(new)]` macro used across the crate; the lint mistakenly
// reports the `#[macro_use]` as unused.
#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;

/// Re-export serde for proc macros.
pub use serde;

/// The configuration module.
pub mod config;

/// Data module.
#[cfg(feature = "std")]
pub mod data;

/// Module for the neural network module.
pub mod module;

/// Module for saving and loading module/optimizer state in the burnpack format.
pub mod store;

/// Module for the tensor.
pub mod tensor;
// Tensor at root: `burn::Tensor`
pub use tensor::Tensor;

#[cfg(feature = "extension")]
/// Backend module.
pub mod backend;

extern crate alloc;

// TODO: configurable device priority
#[cfg(test)]
#[allow(missing_docs)]
pub fn test_device() -> burn_tensor::Device {
    burn_tensor::Device::flex()
}

#[cfg(test)]
mod test_utils {
    use crate as burn;
    use crate::module::Module;
    use crate::module::Param;
    use burn_tensor::Device;
    use burn_tensor::Tensor;

    /// Simple linear module.
    #[derive(Module, Debug)]
    pub struct SimpleLinear {
        pub weight: Param<Tensor<2>>,
        pub bias: Option<Param<Tensor<1>>>,
    }

    impl SimpleLinear {
        pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
            let weight = Tensor::random(
                [out_features, in_features],
                burn_tensor::Distribution::Default,
                device,
            );
            let bias = Tensor::random([out_features], burn_tensor::Distribution::Default, device);

            Self {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        }
    }
}

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! burn::prelude::*` to your code to quickly get started with
    //! Burn.
    pub use crate::{
        config::Config,
        module::Module,
        tensor::{
            Bool, Device, DeviceIndex, DeviceKind, ElementConversion, Float, Int, Shape, SliceArg,
            Tensor, TensorData, assert_shape, cast::ToElement, debug_assert_shape, s,
        },
    };
}
