//! LoRA (Low-Rank Adaptation) Fine-tuning Example
//!
//! This example demonstrates how to use LoRA for parameter-efficient fine-tuning.
//! LoRA freezes the pre-trained model weights and adds small trainable low-rank
//! matrices, reducing the number of trainable parameters significantly.
//!
//! Key features demonstrated:
//! - Applying LoRA to an existing model
//! - Training only LoRA parameters while keeping base weights frozen
//! - Saving and loading LoRA adapters independently of the base model
//! - Merging LoRA weights back into the base model for inference
//!
//! ## Module Structure
//!
//! - [`model`]: Model definitions (SimpleMlp, SimpleMlpWithLora, apply_lora)
//! - [`training`]: Training configuration and execution
//! - [`data`]: Synthetic dataset for demonstration

pub mod data;
pub mod model;
pub mod training;

pub use model::{
    SimpleMlp, SimpleMlpConfig, SimpleMlpWithLora, apply_lora, count_lora_trainable_params,
    count_params,
};
pub use training::{LoraTrainingConfig, run, run_with_config};
