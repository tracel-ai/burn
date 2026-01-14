use burn_core as burn;

use burn::config::Config;
use burn::module::{Initializer, Param};
use burn::tensor::{Tensor, backend::Backend};

/// Bias training mode for LoRA adaptation.
///
/// Controls whether biases are trainable in LoRA-wrapped layers.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum LoraBias {
    /// Keep bias frozen in LoRA-wrapped layers (default).
    None,
    /// Unfreeze bias in LoRA-wrapped layers.
    ///
    /// For non-LoRA layers in the model, manually call `.set_require_grad(true)`
    /// on their biases if you want them to be trainable as well.
    All,
}

/// Weight initialization strategy for LoRA matrices.
///
/// Standard LoRA uses Kaiming initialization for A and zeros for B,
/// ensuring the initial LoRA output is zero (identity behavior).
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum LoraInit {
    /// A = KaimingUniform, B = Zeros (standard LoRA, identity start).
    Kaiming,
    /// Both A and B ~ N(0, 1/sqrt(rank)).
    Gaussian,
    /// Both A and B = Zeros (for testing).
    Zeros,
}

/// Configuration for LoRA (Low-Rank Adaptation) layers.
///
/// LoRA adds trainable low-rank matrices A and B to frozen base weights,
/// computing: `output = base(x) + (x @ A @ B) * scaling`
///
/// The scaling factor is computed as `alpha / rank` (or `alpha / sqrt(rank)` for RSLoRA).
///
/// # Example
///
/// ```ignore
/// use burn_nn::lora::{LoraConfig, LoraAdaptable};
///
/// // Basic usage with rank 16
/// let config = LoraConfig::new(16);
/// let lora_linear = linear_layer.with_lora(&config, &device);
///
/// // Customized configuration
/// let config = LoraConfig::new(32)
///     .with_alpha(64.0)       // scaling = 64/32 = 2.0
///     .with_dropout(0.1)
///     .with_bias(LoraBias::All)
///     .with_use_rslora(true); // scaling = 64/sqrt(32)
/// ```
#[derive(Config, Debug)]
pub struct LoraConfig {
    /// Low-rank dimension (typically 4-64).
    ///
    /// Lower ranks reduce trainable parameters but may limit expressiveness.
    /// Common values: 4, 8, 16, 32, 64.
    pub rank: usize,

    /// Scaling numerator. The scaling factor is `alpha / rank`.
    ///
    /// When `alpha == rank`, the scaling factor is 1.0.
    /// Default is 1.0, giving a scaling factor of `1/rank`.
    #[config(default = 1.0)]
    pub alpha: f64,

    /// Dropout probability applied to LoRA branch input (0.0 = disabled).
    ///
    /// Only active during training (when autodiff is enabled).
    #[config(default = 0.0)]
    pub dropout: f64,

    /// Bias training mode.
    ///
    /// Controls whether biases in LoRA-wrapped layers are trainable.
    #[config(default = "LoraBias::None")]
    pub bias: LoraBias,

    /// Use rank-stabilized LoRA scaling: `alpha / sqrt(rank)` instead of `alpha / rank`.
    ///
    /// RSLoRA maintains more stable gradients across different rank values.
    #[config(default = false)]
    pub use_rslora: bool,

    /// Weight initialization strategy for LoRA matrices A and B.
    #[config(default = "LoraInit::Kaiming")]
    pub init: LoraInit,
}

impl LoraConfig {
    /// Validate the configuration.
    ///
    /// # Panics
    ///
    /// - Panics if `rank` is 0. LoRA requires at least rank 1 to create
    ///   valid low-rank matrices.
    /// - Panics if `dropout` is negative or greater than 1.0.
    pub fn validate(&self) {
        assert!(self.rank > 0, "LoRA rank must be at least 1");
        assert!(
            self.dropout >= 0.0 && self.dropout <= 1.0,
            "LoRA dropout must be between 0.0 and 1.0, got {}",
            self.dropout
        );
    }

    /// Compute the LoRA scaling factor.
    ///
    /// Returns `alpha / rank` normally, or `alpha / sqrt(rank)` if RSLoRA is enabled.
    pub fn scaling(&self) -> f64 {
        if self.use_rslora {
            self.alpha / (self.rank as f64).sqrt()
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Initialize the LoRA A matrix with shape `[in_features, rank]`.
    ///
    /// This is the "down projection" matrix that reduces dimensionality.
    pub fn init_a<B: Backend>(
        &self,
        in_features: usize,
        device: &B::Device,
    ) -> Param<Tensor<B, 2>> {
        let shape = [in_features, self.rank];
        match self.init {
            LoraInit::Kaiming => Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            }
            .init_with(shape, Some(in_features), Some(self.rank), device),
            LoraInit::Gaussian => Initializer::Normal {
                mean: 0.0,
                std: 1.0 / (self.rank as f64).sqrt(),
            }
            .init(shape, device),
            LoraInit::Zeros => Initializer::Zeros.init(shape, device),
        }
    }

    /// Initialize the LoRA B matrix with shape `[rank, out_features]`.
    ///
    /// This is the "up projection" matrix that restores dimensionality.
    /// For Kaiming init, this is zeros so initial LoRA output is zero.
    pub fn init_b<B: Backend>(
        &self,
        out_features: usize,
        device: &B::Device,
    ) -> Param<Tensor<B, 2>> {
        let shape = [self.rank, out_features];
        match self.init {
            // B is zeros for identity start (standard LoRA behavior)
            LoraInit::Kaiming | LoraInit::Zeros => Initializer::Zeros.init(shape, device),
            LoraInit::Gaussian => Initializer::Normal {
                mean: 0.0,
                std: 1.0 / (self.rank as f64).sqrt(),
            }
            .init(shape, device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_standard() {
        let config = LoraConfig::new(8).with_alpha(16.0);
        assert!((config.scaling() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_rslora() {
        let config = LoraConfig::new(16).with_alpha(16.0).with_use_rslora(true);
        // scaling = 16 / sqrt(16) = 16 / 4 = 4.0
        assert!((config.scaling() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_scaling() {
        let config = LoraConfig::new(8);
        // Default alpha=1.0, scaling = 1/8 = 0.125
        assert!((config.scaling() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_config_defaults() {
        let config = LoraConfig::new(16);
        assert_eq!(config.rank, 16);
        assert!((config.alpha - 1.0).abs() < 1e-10);
        assert!((config.dropout - 0.0).abs() < 1e-10);
        assert_eq!(config.bias, LoraBias::None);
        assert!(!config.use_rslora);
        assert_eq!(config.init, LoraInit::Kaiming);
    }

    #[test]
    #[should_panic(expected = "LoRA rank must be at least 1")]
    fn test_rank_zero_panics() {
        // Manually construct config with rank=0 to bypass the normal new() method
        let config = LoraConfig {
            rank: 0,
            alpha: 1.0,
            dropout: 0.0,
            bias: LoraBias::None,
            use_rslora: false,
            init: LoraInit::Kaiming,
        };
        // Validation should panic
        config.validate();
    }

    #[test]
    #[should_panic(expected = "LoRA dropout must be between 0.0 and 1.0")]
    fn test_dropout_negative_panics() {
        let config = LoraConfig {
            rank: 8,
            alpha: 1.0,
            dropout: -0.1,
            bias: LoraBias::None,
            use_rslora: false,
            init: LoraInit::Kaiming,
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "LoRA dropout must be between 0.0 and 1.0")]
    fn test_dropout_greater_than_one_panics() {
        let config = LoraConfig {
            rank: 8,
            alpha: 1.0,
            dropout: 1.5,
            bias: LoraBias::None,
            use_rslora: false,
            init: LoraInit::Kaiming,
        };
        config.validate();
    }

    #[test]
    fn test_dropout_boundary_values_valid() {
        // dropout = 0.0 should be valid
        let config_zero = LoraConfig::new(8).with_dropout(0.0);
        config_zero.validate(); // Should not panic

        // dropout = 1.0 should be valid
        let config_one = LoraConfig::new(8).with_dropout(1.0);
        config_one.validate(); // Should not panic
    }
}
