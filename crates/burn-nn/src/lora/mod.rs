//! LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
//!
//! LoRA enables efficient fine-tuning of large pre-trained models by freezing
//! the original weights and adding trainable low-rank decomposition matrices.
//! This dramatically reduces the number of trainable parameters while maintaining
//! model quality.
//!
//! # Overview
//!
//! Instead of fine-tuning all parameters, LoRA adds pairs of low-rank matrices
//! (A and B) to existing layers. During forward pass:
//!
//! ```text
//! output = base(x) + (x @ A @ B) * scaling
//! ```
//!
//! Where:
//! - `base(x)` is the frozen original layer
//! - `A` has shape `[in_features, rank]` (down projection)
//! - `B` has shape `[rank, out_features]` (up projection)
//! - `scaling = alpha / rank` (configurable)
//!
//! # Usage
//!
//! ```ignore
//! use burn_nn::{Linear, LinearConfig};
//! use burn_nn::lora::{LoraConfig, LoraAdaptable, LoraBias};
//!
//! // Create a pre-trained linear layer
//! let linear = LinearConfig::new(768, 768).init(&device);
//!
//! // Wrap with LoRA
//! let config = LoraConfig::new(16)
//!     .with_alpha(32.0)
//!     .with_dropout(0.1);
//! let lora_linear = linear.with_lora(&config, &device);
//!
//! // Use normally - only LoRA params are trainable
//! let output = lora_linear.forward(input);
//!
//! // Merge for inference (no overhead)
//! let merged = lora_linear.merge();
//! ```
//!
//! # Supported Layers
//!
//! Currently supported:
//! - [`Linear`](crate::Linear) via [`LoraLinear`]
//!
//! Future support planned for Conv1d, Conv2d, Conv3d, Embedding, etc.

mod adapter;
mod config;
mod linear;

pub use adapter::LoraLinearAdapter;
pub use config::{LoraBias, LoraConfig, LoraInit};
pub use linear::{LoraError, LoraLinear};

use burn_core as burn;

use burn::module::Module;
use burn::tensor::backend::Backend;

use crate::lora::config::LoraConfig as Config;

/// Trait for layers that can be wrapped with LoRA adaptation.
///
/// Implementors define how to extract dimensions for LoRA matrices and
/// construct the wrapped version. The wrapper type handles the forward
/// pass and merge logic.
///
/// # Implementing for New Layers
///
/// To add LoRA support for a new layer type:
///
/// 1. Create a wrapper struct (e.g., `LoraConv2d`)
/// 2. Implement `LoraAdaptable` for the base layer
/// 3. Implement `forward()` and `merge()` on the wrapper
///
/// ```ignore
/// impl<B: Backend> LoraAdaptable<B> for Conv2d<B> {
///     type Wrapped = LoraConv2d<B>;
///
///     fn lora_dims(&self) -> (usize, usize) {
///         let [out_ch, in_ch, kh, kw] = self.weight.shape().dims();
///         (in_ch * kh * kw, out_ch)
///     }
///
///     fn with_lora(self, config: &LoraConfig, device: &B::Device) -> Self::Wrapped {
///         // ... implementation
///     }
/// }
/// ```
pub trait LoraAdaptable<B: Backend>: Module<B> + Sized {
    /// The LoRA-wrapped version of this layer.
    ///
    /// This associated type ensures type safety - each base layer
    /// maps to exactly one wrapper type.
    type Wrapped: Module<B>;

    /// Returns dimensions for LoRA matrices: `(in_features, out_features)`.
    fn lora_dims(&self) -> (usize, usize);

    /// Wrap this layer with LoRA adaptation.
    ///
    /// This method:
    /// - Consumes `self` (the original layer)
    /// - Freezes the base weights
    /// - Initializes LoRA matrices A and B
    /// - Optionally unfreezes bias based on config
    fn with_lora(self, config: &Config, device: &B::Device) -> Self::Wrapped;
}
