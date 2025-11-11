use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend, module::linear};

use crate::ops::merge_lora;

/// Configuration for creating a [`LoRAPlusLinear`] layer.
///
/// LoRA+ improves upon LoRA by using different learning rates for A and B matrices.
///
/// # Key Insight
///
/// The paper "LoRA+: Efficient Low Rank Adaptation of Large Models" shows that
/// setting a higher learning rate for B (typically 16x higher than A) leads to:
/// - Faster convergence
/// - Better final performance
/// - More stable training
///
/// # Recommended Learning Rates
///
/// If base learning rate is η:
/// - **Learning rate for A**: η
/// - **Learning rate for B**: 16η (set via optimizer groups)
///
/// # Example
///
/// ```rust,ignore
/// use burn_peft::{LoRAPlusConfig, LoRAPlusLinear};
///
/// let config = LoRAPlusConfig::new(512, 512)
///     .with_rank(16)
///     .with_alpha(32.0)
///     .with_lr_ratio(16.0);  // B will have 16x learning rate of A
///
/// let layer = config.init::<MyBackend>(&device);
///
/// // In your optimizer, set different learning rates:
/// // - base_lr for layer.lora_a
/// // - base_lr * 16.0 for layer.lora_b
/// ```
///
/// # Reference
///
/// Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", 2024
/// https://arxiv.org/abs/2402.12354
#[derive(Config, Debug)]
pub struct LoRAPlusConfig {
    /// Input dimension
    pub d_input: usize,
    /// Output dimension
    pub d_output: usize,
    /// Rank of the low-rank adaptation (r)
    #[config(default = 8)]
    pub rank: usize,
    /// Scaling factor alpha (α)
    #[config(default = 8.0)]
    pub alpha: f64,
    /// Learning rate ratio: lr(B) / lr(A)
    /// Recommended: 16.0
    #[config(default = 16.0)]
    pub lr_ratio: f64,
    /// Dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether to include bias
    #[config(default = true)]
    pub bias: bool,
    /// Initializer for matrix A
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer_a: Initializer,
    /// Initializer for matrix B (default: zeros)
    #[config(default = "Initializer::Zeros")]
    pub initializer_b: Initializer,
    /// Initializer for base weight W₀
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer_base: Initializer,
}

impl LoRAPlusConfig {
    /// Initialize from pretrained weight
    pub fn init_with_base_weight<B: Backend>(
        &self,
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> LoRAPlusLinear<B> {
        let weight = Param::from_tensor(weight);
        let bias = bias.map(Param::from_tensor);

        let lora_a = self.initializer_a.init_with(
            [self.rank, self.d_input],
            Some(self.rank),
            Some(self.d_input),
            device,
        );

        let lora_b = self.initializer_b.init_with(
            [self.d_output, self.rank],
            Some(self.d_output),
            Some(self.rank),
            device,
        );

        LoRAPlusLinear {
            weight,
            bias,
            lora_a,
            lora_b,
            alpha: self.alpha,
            rank: self.rank,
            dropout: self.dropout,
            lr_ratio: self.lr_ratio,
            merged: false,
        }
    }
}

/// LoRA+ linear layer with optimized learning rates.
///
/// This is identical to LoRA in forward pass and structure, but is designed
/// to be used with different learning rates for A and B matrices.
///
/// # Training Tips
///
/// **When creating your optimizer, set different learning rates:**
///
/// ```rust,ignore
/// // Pseudo-code (Burn doesn't have per-parameter lr yet)
/// // This is the intended usage pattern:
///
/// let base_lr = 1e-3;
/// let optimizer = AdamConfig::new()
///     .with_learning_rate(base_lr)
///     .init();
///
/// // For each LoRA+ layer, you'd ideally set:
/// // - lr(lora_a) = base_lr
/// // - lr(lora_b) = base_lr * lr_ratio (typically 16x)
///
/// // Note: Burn's optimizer API is evolving to support this.
/// // For now, you can approximate this by using a higher global learning rate
/// // and scaling A's gradients down during training.
/// ```
///
/// # Why Does This Work?
///
/// The LoRA+ paper shows that B needs to adapt faster than A because:
/// 1. B starts at zero (so needs larger updates)
/// 2. A is responsible for feature selection
/// 3. B is responsible for feature mixing
///
/// Different learning rates let each matrix learn at its optimal pace.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LoRAPlusLinear<B: Backend> {
    /// Base weight W₀ of shape [d_input, d_output]
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// LoRA A matrix of shape [rank, d_input]
    /// Should use base learning rate
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix of shape [d_output, rank]
    /// Should use lr_ratio × base learning rate
    pub lora_b: Param<Tensor<B, 2>>,
    /// Scaling factor α
    pub alpha: f64,
    /// Rank r
    pub rank: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Recommended learning rate ratio (lr_B / lr_A)
    #[module(skip)]
    pub lr_ratio: f64,
    /// Whether adapters are currently merged
    #[module(skip)]
    pub merged: bool,
}

impl<B: Backend> LoRAPlusLinear<B> {
    /// Forward pass (identical to LoRA)
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if self.merged {
            linear(
                input,
                self.weight.val(),
                self.bias.as_ref().map(|b| b.val()),
            )
        } else {
            let base_out = linear(
                input.clone(),
                self.weight.val(),
                self.bias.as_ref().map(|b| b.val()),
            );

            let alpha_over_r = (self.alpha / self.rank as f64) as f32;

            let a_out = linear(input, self.lora_a.val().transpose(), None);
            let delta = linear(a_out, self.lora_b.val().transpose(), None);

            base_out + delta * alpha_over_r
        }
    }

    /// Merge adapters (same as LoRA)
    pub fn merge_weights(&mut self) {
        if !self.merged {
            let alpha_over_r = (self.alpha / self.rank as f64) as f32;
            let merged_weight = merge_lora(
                &self.weight.val(),
                &self.lora_b.val(),
                &self.lora_a.val(),
                alpha_over_r,
            );
            self.weight = Param::from_tensor(merged_weight);
            self.merged = true;
        }
    }

    /// Unmerge adapters (same as LoRA)
    pub fn unmerge_weights(&mut self) {
        if self.merged {
            let alpha_over_r = (self.alpha / self.rank as f64) as f32;
            let base_weight = crate::ops::unmerge_lora(
                &self.weight.val(),
                &self.lora_b.val(),
                &self.lora_a.val(),
                alpha_over_r,
            );
            self.weight = Param::from_tensor(base_weight);
            self.merged = false;
        }
    }

    /// Get recommended learning rate ratio
    pub fn get_lr_ratio(&self) -> f64 {
        self.lr_ratio
    }
}

impl<B: Backend> ModuleDisplay for LoRAPlusLinear<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.weight.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("rank", &self.rank)
            .add("alpha", &self.alpha)
            .add("lr_ratio", &self.lr_ratio)
            .add("bias", &self.bias.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Distribution;

    #[test]
    fn test_lora_plus_forward() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let weight = Tensor::<TestBackend, 2>::random([32, 64], Distribution::Default, &device);
        let config = LoRAPlusConfig::new(32, 64).with_rank(8).with_lr_ratio(16.0);
        let layer = config.init_with_base_weight(weight, None, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [4, 64]);
        assert_eq!(layer.get_lr_ratio(), 16.0);
    }

    #[test]
    fn display() {
        let device = Default::default();
        let weight = Tensor::<TestBackend, 2>::zeros([128, 256], &device);
        let config = LoRAPlusConfig::new(128, 256)
            .with_rank(16)
            .with_lr_ratio(16.0);
        let layer = config.init_with_base_weight(weight, None, &device);

        let display = alloc::format!("{layer}");
        assert!(display.contains("d_input: 128"));
        assert!(display.contains("d_output: 256"));
        assert!(display.contains("lr_ratio: 16"));
    }
}
