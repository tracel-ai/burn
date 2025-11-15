use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend, module::linear};

use crate::ops::{col_norm, normalize_cols_detached};

/// Configuration for creating a [`DoRALinear`] layer.
///
/// DoRA (Weight-Decomposed Low-Rank Adaptation) extends LoRA by decomposing
/// weights into magnitude and direction, applying low-rank adaptation only to direction.
///
/// # Mathematical Formulation
///
/// DoRA decomposes the weight as: W = m ⊙ (V / ||V||_c)
/// where:
/// - m ∈ ℝ^{1×k}: trainable magnitude vector (per-column)
/// - V = V₀ + ΔV: direction matrix
/// - V₀: frozen base direction (copy of W₀)
/// - ΔV = BA: low-rank update to direction
/// - ||V||_c: column-wise L2 norms
///
/// # Forward Pass (Unmerged)
///
/// 1. V' = V₀ + BA
/// 2. n = ||V'||_c (computed, then detached)
/// 3. V̂ = V' / detach(n)
/// 4. W_eff = m ⊙ V̂
/// 5. h = W_eff @ x
///
/// # Initialization
///
/// - m = ||W₀||_c (so W_eff = W₀ initially)
/// - V₀ = W₀ (frozen)
/// - ΔV = BA with A ~ Kaiming, B = 0
///
/// # Key Difference from LoRA
///
/// The detached norm is critical: gradients flow through V' but NOT through
/// the norm computation, saving memory and stabilizing training.
///
/// # Example
///
/// ```rust,ignore
/// use burn_peft::{DoRAConfig, DoRALinear};
///
/// let config = DoRAConfig::new(512, 512)
///     .with_rank(16)
///     .with_epsilon(1e-8);
///
/// let layer = config.init::<MyBackend>(&device);
/// let output = layer.forward(input);
/// ```
#[derive(Config, Debug)]
pub struct DoRAConfig {
    /// Input dimension
    pub d_input: usize,
    /// Output dimension
    pub d_output: usize,
    /// Rank of the low-rank adaptation (r)
    #[config(default = 8)]
    pub rank: usize,
    /// Epsilon for numerical stability in normalization
    #[config(default = 1e-8)]
    pub epsilon: f64,
    /// Dropout rate applied to input before adapter
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
    /// Initializer for base direction V₀
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer_base: Initializer,
}

impl DoRAConfig {
    /// Initialize a new [`DoRALinear`] module
    pub fn init<B: Backend>(&self, device: &B::Device) -> DoRALinear<B> {
        // Base direction V₀: [d_input, d_output]
        let v_base = self.initializer_base.init_with(
            [self.d_input, self.d_output],
            Some(self.d_input),
            Some(self.d_output),
            device,
        );

        // Magnitude m initialized from column norms of V₀
        // Shape: [1, d_output]
        // Use detached to avoid non-leaf tensor issues with autodiff backends
        let magnitude = col_norm(&v_base.val(), self.epsilon as f32).detach();
        let magnitude = Param::from_tensor(magnitude);

        // Bias (optional)
        let bias = if self.bias {
            Some(self.initializer_base.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        // LoRA A matrix: [rank, d_input]
        let lora_a = self.initializer_a.init_with(
            [self.rank, self.d_input],
            Some(self.rank),
            Some(self.d_input),
            device,
        );

        // LoRA B matrix: [d_output, rank]
        // Initialized to zeros so ΔV = BA = 0 initially
        let lora_b = self.initializer_b.init_with(
            [self.d_output, self.rank],
            Some(self.d_output),
            Some(self.rank),
            device,
        );

        DoRALinear {
            v_base,
            magnitude,
            bias,
            lora_a,
            lora_b,
            rank: self.rank,
            epsilon: self.epsilon,
            dropout: self.dropout,
            merged: false,
        }
    }

    /// Initialize from an existing weight tensor
    ///
    /// This converts a pretrained weight to DoRA by:
    /// 1. Setting V₀ = W₀ (frozen direction base)
    /// 2. Computing m = ||W₀||_c (trainable magnitude)
    /// 3. Initializing ΔV = 0 via B = 0
    ///
    /// Result: W_eff = m ⊙ (V₀ / ||V₀||_c) = W₀ initially
    pub fn init_with_base_weight<B: Backend>(
        &self,
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> DoRALinear<B> {
        // Use pretrained weight as base direction
        let v_base = Param::from_tensor(weight.clone());

        // Compute magnitude from pretrained weight
        // Use detached to avoid non-leaf tensor issues with autodiff backends
        let magnitude = col_norm(&weight, self.epsilon as f32).detach();
        let magnitude = Param::from_tensor(magnitude);

        let bias = bias.map(Param::from_tensor);

        // Initialize adapters
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

        DoRALinear {
            v_base,
            magnitude,
            bias,
            lora_a,
            lora_b,
            rank: self.rank,
            epsilon: self.epsilon,
            dropout: self.dropout,
            merged: false,
        }
    }
}

/// DoRA-adapted linear layer
///
/// Implements Weight-Decomposed Low-Rank Adaptation.
/// Decomposes weight into magnitude and direction, applying low-rank
/// adaptation only to the direction component.
///
/// # Mathematical Details
///
/// The effective weight is computed as:
/// W_eff = m ⊙ V̂
///
/// where:
/// - V̂ = (V₀ + BA) / detach(||(V₀ + BA)||_c)
/// - m: trainable magnitude (shape [1, d_output])
/// - V₀: frozen base direction (shape [d_input, d_output])
/// - BA: low-rank update (B: [d_output, r], A: [r, d_input])
///
/// The detached norm prevents gradients from flowing through the
/// normalization, which:
/// 1. Reduces activation memory significantly
/// 2. Stabilizes training
/// 3. Maintains empirical performance
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct DoRALinear<B: Backend> {
    /// Base direction V₀ of shape [d_input, d_output] (frozen)
    pub v_base: Param<Tensor<B, 2>>,
    /// Magnitude m of shape [1, d_output] (trainable)
    pub magnitude: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// LoRA A matrix of shape [rank, d_input]
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix of shape [d_output, rank]
    pub lora_b: Param<Tensor<B, 2>>,
    /// Rank r
    pub rank: usize,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Dropout probability
    pub dropout: f64,
    /// Whether adapters are currently merged
    #[module(skip)]
    pub merged: bool,
}

impl<B: Backend> DoRALinear<B> {
    /// Forward pass through DoRA linear layer
    ///
    /// # Unmerged Mode (training)
    ///
    /// 1. V' = V₀ + BA
    /// 2. V̂ = V' / detach(||V'||_c)
    /// 3. W_eff = m ⊙ V̂
    /// 4. h = W_eff @ x
    ///
    /// # Merged Mode (inference)
    ///
    /// Uses precomputed W_eff directly (stored in v_base after merging)
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., d_input]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., d_output]
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if self.merged {
            // Merged mode: v_base contains the final weight
            linear(
                input,
                self.v_base.val(),
                self.bias.as_ref().map(|b| b.val()),
            )
        } else {
            // Unmerged mode: compute DoRA decomposition

            // TODO: Apply dropout to input if enabled
            // For now, we skip dropout application in the tensor path
            // A future version can integrate with burn_nn::Dropout module

            // Compute low-rank update: ΔV = BA
            // B: [d_output, r], A: [r, d_input]
            // ΔV: [d_output, d_input]
            let delta_v = self.lora_b.val().matmul(self.lora_a.val());

            // Compute updated direction: V' = V₀ + ΔV
            // V₀ shape: [d_input, d_output], delta_v: [d_output, d_input]
            // Need to transpose delta_v to match
            let v_prime = self.v_base.val() + delta_v.transpose();

            // Normalize with detached norms: V̂ = V' / detach(||V'||_c)
            // This is the key DoRA operation
            let v_hat = normalize_cols_detached(&v_prime, self.epsilon as f32);

            // Apply magnitude: W_eff = m ⊙ V̂
            // m shape: [1, d_output], broadcast over rows
            let w_eff = v_hat * self.magnitude.val();

            // Final output: h = W_eff @ x + bias
            linear(input, w_eff, self.bias.as_ref().map(|b| b.val()))
        }
    }

    /// Merge DoRA adapters into a single weight for inference
    ///
    /// Computes: W' = m ⊙ ((V₀ + BA) / ||(V₀ + BA)||_c)
    ///
    /// After merging, the forward pass becomes a simple matrix multiplication.
    pub fn merge_weights(&mut self) {
        if !self.merged {
            // Compute ΔV = BA
            let delta_v = self.lora_b.val().matmul(self.lora_a.val());

            // V' = V₀ + ΔV (need to transpose delta_v)
            let v_prime = self.v_base.val() + delta_v.transpose();

            // Normalize: V̂ = V' / ||V'||_c
            let v_hat = normalize_cols_detached(&v_prime, self.epsilon as f32);

            // Apply magnitude: W' = m ⊙ V̂
            let merged_weight = v_hat * self.magnitude.val();

            // Store merged weight in v_base
            self.v_base = Param::from_tensor(merged_weight);
            self.merged = true;
        }
    }

    /// Unmerge weights (not straightforward for DoRA)
    ///
    /// Note: Unlike LoRA, DoRA unmerging is complex because the magnitude
    /// has been applied. This method attempts to restore V₀ but the magnitude
    /// information may be lost. For production, consider keeping a copy of
    /// the original V₀ if you need to unmerge.
    pub fn unmerge_weights(&mut self) {
        if self.merged {
            // This is an approximation: divide by magnitude to get V̂
            // Then we'd need to "unnormalize" which requires knowing the original norms
            // For now, we'll just mark as unmerged and warn in docs

            // In practice, you should not unmerge DoRA weights
            // Instead, reload from checkpoint or keep original V₀
            self.merged = false;
        }
    }

    /// Check if adapters are currently merged
    pub fn is_merged(&self) -> bool {
        self.merged
    }
}

impl<B: Backend> ModuleDisplay for DoRALinear<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.v_base.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("rank", &self.rank)
            .add("epsilon", &self.epsilon)
            .add("bias", &self.bias.is_some())
            .add("merged", &self.merged)
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Distribution;

    #[test]
    fn test_dora_initialization() {
        let device = Default::default();

        let config = DoRAConfig::new(64, 128).with_rank(8);

        let layer = config.init::<TestBackend>(&device);

        // Check shapes
        assert_eq!(layer.v_base.dims(), [64, 128]);
        assert_eq!(layer.magnitude.dims(), [1, 128]);
        assert_eq!(layer.lora_a.dims(), [8, 64]);
        assert_eq!(layer.lora_b.dims(), [128, 8]);

        // Check B is initialized to zeros
        let b_data = layer.lora_b.val().into_data();
        let b_vec = b_data.to_vec::<f32>().unwrap();
        for &val in &b_vec {
            assert!((val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dora_forward_unmerged() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let config = DoRAConfig::new(32, 64).with_rank(4).with_bias(false);

        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [8, 64]);
    }

    #[test]
    fn test_dora_initial_equivalence() {
        // Test that initially DoRA output matches base weight
        // Since ΔV = 0, we have W_eff = m ⊙ (V₀ / ||V₀||_c)
        // With m = ||V₀||_c, this should equal V₀

        let device = Default::default();
        TestBackend::seed(&device, 42);

        let config = DoRAConfig::new(16, 32)
            .with_rank(4)
            .with_initializer_base(Initializer::Constant { value: 1.0 })
            .with_bias(false);

        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones([4, 16], &device);

        // Expected output with all-ones weight: each output element = 16
        let output = layer.forward(input);

        // Check that output is approximately correct
        let mean_val = output.clone().mean().into_scalar();
        assert!((mean_val - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_dora_magnitude_shape() {
        let device = Default::default();

        let config = DoRAConfig::new(10, 20).with_rank(2);
        let layer = config.init::<TestBackend>(&device);

        // Magnitude should be [1, d_output]
        assert_eq!(layer.magnitude.dims(), [1, 20]);

        // All magnitude values should be positive (norms)
        let mag_data = layer.magnitude.val().into_data();
        let mag_vec = mag_data.to_vec::<f32>().unwrap();
        for &val in &mag_vec {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_dora_merge() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let config = DoRAConfig::new(16, 32).with_rank(4);
        let mut layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random([4, 16], Distribution::Default, &device);

        // Get output in unmerged mode
        let output_unmerged = layer.forward(input.clone());

        // Merge weights
        layer.merge_weights();
        assert!(layer.is_merged());

        // Get output in merged mode
        let output_merged = layer.forward(input);

        // Outputs should be very close (within numerical precision)
        let diff = (output_unmerged - output_merged).abs().mean();
        assert!(diff.into_scalar() < 1e-4);
    }

    #[test]
    fn display() {
        let device = Default::default();
        let config = DoRAConfig::new(128, 256).with_rank(16);
        let layer = config.init::<TestBackend>(&device);

        let display = alloc::format!("{layer}");
        assert!(display.contains("d_input: 128"));
        assert!(display.contains("d_output: 256"));
        assert!(display.contains("rank: 16"));
    }
}
