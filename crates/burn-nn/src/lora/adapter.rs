//! LoRA adapter persistence for saving and loading adapter weights.
//!
//! This module provides `LoraLinearAdapter` for saving and loading LoRA adapter weights
//! independently of the base model. This enables:
//!
//! - **Efficient storage**: Save only the small LoRA matrices (A & B), not the entire model
//! - **Adapter swapping**: Load different adapters onto the same base model at runtime
//! - **Portability**: Share adapter weights separately from base model weights
//!
//! # File Format
//!
//! Adapters are saved as MessagePack (`.mpk`) files containing:
//! - `config`: LoRA configuration (rank, alpha, dropout, etc.)
//! - `lora_a`: Down-projection matrix `[d_input, rank]`
//! - `lora_b`: Up-projection matrix `[rank, d_output]`
//!
//! # Example
//!
//! ```ignore
//! use burn::record::NamedMpkFileRecorder;
//! use burn::record::FullPrecisionSettings;
//!
//! let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
//!
//! // Save adapter after training
//! lora_linear.save_adapter("./my-adapter", &recorder)?;
//!
//! // Load adapter onto fresh LoRA layer
//! let loaded = fresh_lora_linear.load_adapter_file("./my-adapter", &recorder, &device)?;
//! ```

use burn_core as burn;

use burn::module::Param;
use burn::record::{FloatTensorSerde, ParamSerde, PrecisionSettings, Record};
use burn::serde::{Deserialize, Serialize};
use burn::tensor::{Tensor, backend::Backend};

use super::config::LoraConfig;

/// Standalone LoRA adapter record containing only trainable weights.
///
/// This struct captures the essential LoRA weights (A and B matrices) along with
/// the configuration needed to reconstruct the adapter. It does NOT include base
/// model weights - those are loaded separately.
///
/// Use this for:
/// - Saving trained LoRA weights independently of the base model
/// - Loading adapters onto different base models
/// - Swapping adapters at runtime without reloading the base model
///
/// # Fields
///
/// - `lora_a`: Down-projection matrix with shape `[d_input, rank]`
/// - `lora_b`: Up-projection matrix with shape `[rank, d_output]`
/// - `config`: LoRA configuration (rank, alpha, dropout, bias, etc.)
#[derive(Debug, Clone)]
pub struct LoraLinearAdapter<B: Backend> {
    /// LoRA A matrix (down-projection): `[d_input, rank]`.
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix (up-projection): `[rank, d_output]`.
    pub lora_b: Param<Tensor<B, 2>>,
    /// LoRA configuration used to create this adapter.
    pub config: LoraConfig,
}

/// Serializable form of `LoraLinearAdapter` for persistence.
///
/// This is the `Item` type used by the `Record` trait implementation.
/// It uses `ParamSerde` for tensor serialization with precision settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "burn::serde", bound = "")]
pub struct LoraLinearAdapterItem<S: PrecisionSettings> {
    /// Serialized LoRA A matrix.
    pub lora_a: ParamSerde<FloatTensorSerde<S>>,
    /// Serialized LoRA B matrix.
    pub lora_b: ParamSerde<FloatTensorSerde<S>>,
    /// LoRA configuration.
    pub config: LoraConfig,
}

impl<B: Backend> Record<B> for LoraLinearAdapter<B> {
    type Item<S: PrecisionSettings> = LoraLinearAdapterItem<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        LoraLinearAdapterItem {
            lora_a: self.lora_a.into_item(),
            lora_b: self.lora_b.into_item(),
            config: self.config,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        LoraLinearAdapter {
            lora_a: Param::from_item(item.lora_a, device),
            lora_b: Param::from_item(item.lora_b, device),
            config: item.config,
        }
    }
}

impl<B: Backend> LoraLinearAdapter<B> {
    /// Creates a new adapter from LoRA matrices and configuration.
    pub fn new(
        lora_a: Param<Tensor<B, 2>>,
        lora_b: Param<Tensor<B, 2>>,
        config: LoraConfig,
    ) -> Self {
        Self {
            lora_a,
            lora_b,
            config,
        }
    }

    /// Returns the LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.shape().dims::<2>()[1]
    }

    /// Returns the input dimension.
    pub fn d_input(&self) -> usize {
        self.lora_a.shape().dims::<2>()[0]
    }

    /// Returns the output dimension.
    pub fn d_output(&self) -> usize {
        self.lora_b.shape().dims::<2>()[1]
    }
}
