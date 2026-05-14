#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "135"]

//! The core crate of Burn.

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

/// Module for the recorder.
pub mod record;

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
pub type TestDevice = burn_tensor::NdArrayDevice;

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
            Bool, Device, ElementConversion, Float, Int, Shape, SliceArg, Tensor, TensorData,
            cast::ToElement, s,
        },
    };
    pub use burn_std::device::Device as DeviceOps;
}
