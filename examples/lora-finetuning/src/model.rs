//! LoRA-adapted MLP model for fine-tuning demonstration.
//!
//! This module contains the model definitions:
//! - `SimpleMlp`: Base MLP model that can be adapted with LoRA
//! - `SimpleMlpWithLora`: MLP with LoRA applied to fc1 and fc2 layers
//! - `apply_lora`: Function to apply LoRA to a SimpleMlp model

use burn::nn::lora::{LoraConfig, LoraLinear};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};

use std::path::PathBuf;

/// A simple MLP model that can be adapted with LoRA.
#[derive(Module, Debug)]
pub struct SimpleMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub fc3: Linear<B>,
    pub relu: Relu,
}

/// Configuration for the SimpleMlp model.
#[derive(Config, Debug)]
pub struct SimpleMlpConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub d_output: usize,
}

impl SimpleMlpConfig {
    /// Initialize a new SimpleMlp model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleMlp<B> {
        SimpleMlp {
            fc1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            fc2: LinearConfig::new(self.d_hidden, self.d_hidden).init(device),
            fc3: LinearConfig::new(self.d_hidden, self.d_output).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> SimpleMlp<B> {
    /// Forward pass through the MLP.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        self.fc3.forward(x)
    }
}

/// The same MLP but with LoRA applied to fc1 and fc2.
/// fc3 remains unchanged (common pattern: adapt attention/hidden, keep output fixed).
#[derive(Module, Debug)]
pub struct SimpleMlpWithLora<B: Backend> {
    pub fc1: LoraLinear<B>,
    pub fc2: LoraLinear<B>,
    pub fc3: Linear<B>,
    pub relu: Relu,
}

impl<B: Backend> SimpleMlpWithLora<B> {
    /// Forward pass through the LoRA-adapted MLP.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        self.fc3.forward(x)
    }

    /// Merge LoRA weights into base layers for inference.
    /// Returns a regular SimpleMlp with no LoRA overhead.
    pub fn merge(self) -> SimpleMlp<B> {
        SimpleMlp {
            fc1: self.fc1.merge(),
            fc2: self.fc2.merge(),
            fc3: self.fc3,
            relu: self.relu,
        }
    }

    /// Save adapter weights to disk.
    ///
    /// Saves only the LoRA matrices (not the base model) to `.mpk` files.
    /// This enables efficient storage and adapter swapping.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to save adapters (e.g., "./my-adapter")
    ///
    /// # File Structure
    ///
    /// ```text
    /// my-adapter/
    /// ├── fc1.mpk   # LoRA config + A & B matrices for fc1
    /// └── fc2.mpk   # LoRA config + A & B matrices for fc2
    /// ```
    pub fn save_adapters(&self, path: impl Into<PathBuf>) -> Result<(), RecorderError> {
        let path = path.into();
        std::fs::create_dir_all(&path).map_err(|e| RecorderError::Unknown(e.to_string()))?;

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        self.fc1.save_adapter(path.join("fc1"), &recorder)?;
        self.fc2.save_adapter(path.join("fc2"), &recorder)?;

        Ok(())
    }

    /// Load adapter weights from disk.
    ///
    /// Loads LoRA matrices from `.mpk` files and applies them to this model.
    /// The base model weights remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path containing adapter files
    /// * `device` - Device to load tensors onto
    pub fn load_adapters(
        self,
        path: impl Into<PathBuf>,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let path = path.into();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let fc1 = self
            .fc1
            .load_adapter_file(path.join("fc1"), &recorder, device)?;
        let fc2 = self
            .fc2
            .load_adapter_file(path.join("fc2"), &recorder, device)?;

        Ok(Self {
            fc1,
            fc2,
            fc3: self.fc3,
            relu: self.relu,
        })
    }
}

/// Apply LoRA to a SimpleMlp model.
///
/// This function demonstrates the common pattern of:
/// 1. Taking a pre-trained model
/// 2. Wrapping specific layers with LoRA
/// 3. Returning a new model with frozen base weights and trainable LoRA params
pub fn apply_lora<B: Backend>(
    model: SimpleMlp<B>,
    config: &LoraConfig,
    device: &B::Device,
) -> SimpleMlpWithLora<B> {
    use burn::nn::lora::LoraAdaptable;

    SimpleMlpWithLora {
        fc1: model.fc1.with_lora(config, device),
        fc2: model.fc2.with_lora(config, device),
        fc3: model.fc3.no_grad(), // Keep fc3 frozen without LoRA
        relu: model.relu,
    }
}

/// Count total parameters in a model.
pub fn count_params<B: Backend>(model: &SimpleMlp<B>) -> usize {
    let fc1_params = model.fc1.weight.shape().num_elements()
        + model
            .fc1
            .bias
            .as_ref()
            .map_or(0, |b| b.shape().num_elements());
    let fc2_params = model.fc2.weight.shape().num_elements()
        + model
            .fc2
            .bias
            .as_ref()
            .map_or(0, |b| b.shape().num_elements());
    let fc3_params = model.fc3.weight.shape().num_elements()
        + model
            .fc3
            .bias
            .as_ref()
            .map_or(0, |b| b.shape().num_elements());
    fc1_params + fc2_params + fc3_params
}

/// Count trainable LoRA parameters.
pub fn count_lora_trainable_params<B: Backend>(model: &SimpleMlpWithLora<B>) -> usize {
    // LoRA A and B matrices for fc1 and fc2
    let fc1_lora =
        model.fc1.lora_a.shape().num_elements() + model.fc1.lora_b.shape().num_elements();
    let fc2_lora =
        model.fc2.lora_a.shape().num_elements() + model.fc2.lora_b.shape().num_elements();
    fc1_lora + fc2_lora
}
