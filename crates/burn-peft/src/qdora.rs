use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend, module::linear};

use crate::ops::{col_norm, normalize_cols_detached};

/// Configuration for creating a [`QDoRALinear`] layer.
///
/// QDoRA (Quantized DoRA) combines weight decomposition with quantized base weights
/// for maximum memory efficiency while maintaining DoRA's performance benefits.
///
/// # Mathematical Formulation
///
/// Base direction V₀ is stored quantized:
/// 1. V' = Dequant(V₀_q) + BA (on-the-fly dequantization)
/// 2. n = ||V'||_c (column-wise norms, computed in FP32)
/// 3. V̂ = V' / detach(n) (detached for memory efficiency)
/// 4. W_eff = m ⊙ V̂ (apply trainable magnitude)
/// 5. h = W_eff @ x
///
/// # Memory Savings
///
/// With 4-bit quantization and rank r=8 on a 4096×4096 layer:
/// - Full DoRA: 64 MB direction + 16 KB magnitude + 0.5 MB adapters ≈ 64.5 MB
/// - QDoRA: 8 MB direction + 16 KB magnitude + 0.5 MB adapters ≈ 8.5 MB
///
/// **87% memory reduction!**
///
/// # Example
///
/// ```rust,ignore
/// use burn_peft::{QDoRAConfig, QDoRALinear};
/// use burn::tensor::quantization::{QuantScheme, QuantValue, QuantLevel};
///
/// let scheme = QuantScheme::default()
///     .with_value(QuantValue::Q4F)
///     .with_level(QuantLevel::block([32]));
///
/// let config = QDoRAConfig::new(4096, 4096)
///     .with_rank(8)
///     .with_epsilon(1e-8);
///
/// let layer = config.init::<MyBackend>(&device);
/// ```
#[derive(Config, Debug)]
pub struct QDoRAConfig {
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
}

impl QDoRAConfig {
    /// Initialize from a pretrained quantized weight
    ///
    /// This converts a quantized pretrained weight to QDoRA by:
    /// 1. Dequantizing temporarily to compute magnitude: m = ||Dequant(W₀_q)||_c
    /// 2. Using W₀_q as the quantized base direction V₀
    /// 3. Initializing adapters: A ~ Kaiming, B = 0
    ///
    /// Result: W_eff = m ⊙ (Dequant(V₀_q) / ||Dequant(V₀_q)||_c) = W₀ initially
    ///
    /// # Arguments
    ///
    /// * `weight_quantized` - Pre-quantized base weight
    /// * `bias` - Optional bias
    /// * `device` - Device to create adapters on
    pub fn init_with_quantized_weight<B: Backend>(
        &self,
        weight_quantized: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> QDoRALinear<B> {
        // Temporarily dequantize to compute magnitude
        let weight_fp = weight_quantized.clone().dequantize();
        // Use detached to avoid non-leaf tensor issues with autodiff backends
        let magnitude = col_norm(&weight_fp, self.epsilon as f32).detach();
        let magnitude = Param::from_tensor(magnitude);

        // Use quantized weight as base direction
        let v_base = Param::from_tensor(weight_quantized);

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

        QDoRALinear {
            v_base,
            magnitude,
            bias,
            lora_a,
            lora_b,
            rank: self.rank,
            epsilon: self.epsilon,
            dropout: self.dropout,
        }
    }
}

/// QDoRA-adapted linear layer
///
/// Implements Quantized Weight-Decomposed Low-Rank Adaptation.
///
/// # Architecture
///
/// ```text
/// Input x
///   │
///   ├─> Dequant(V₀_q) ──┐
///   │                    │
///   └─> A @ x ─> B @ (·) ┴─> V' = Dequant(V₀_q) + BA
///                              │
///                              ├─> norms = ||V'||_c (FP32)
///                              │
///                              ├─> V̂ = V' / detach(norms)
///                              │
///                              └─> W_eff = m ⊙ V̂
///                                    │
///                                    └─> W_eff @ x ──> Output
/// ```
///
/// # Memory Efficiency
///
/// The quantized base direction uses 4–8 bits per parameter. The magnitude
/// vector and adapters are tiny compared to the base (e.g., for a 4096×4096
/// layer, magnitude is 16KB and adapters are ~500KB at rank 8).
///
/// # Training
///
/// Trainable: `magnitude` (m), `lora_a` (A), `lora_b` (B)
/// Frozen: `v_base` (quantized V₀)
///
/// The detached norm prevents gradients from flowing through the normalization,
/// which is crucial for memory efficiency and training stability.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct QDoRALinear<B: Backend> {
    /// Quantized base direction V₀_q of shape [d_input, d_output] (frozen)
    pub v_base: Param<Tensor<B, 2>>,
    /// Magnitude m of shape [1, d_output] (trainable)
    pub magnitude: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// LoRA A matrix of shape [rank, d_input] (trainable)
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix of shape [d_output, rank] (trainable)
    pub lora_b: Param<Tensor<B, 2>>,
    /// Rank r
    pub rank: usize,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Dropout probability
    pub dropout: f64,
}

impl<B: Backend> QDoRALinear<B> {
    /// Forward pass through QDoRA linear layer
    ///
    /// # Computation Steps
    ///
    /// 1. **Dequantize base:** V₀ = Dequant(V₀_q)
    ///    - On-the-fly dequantization (may use fused kernels)
    ///
    /// 2. **Compute direction update:** V' = V₀ + BA
    ///    - Low-rank path computed in full precision
    ///
    /// 3. **Normalize with detached norms:**
    ///    - n = ||V'||_c (column-wise L2 norms in FP32)
    ///    - V̂ = V' / detach(n) (gradients don't flow through n)
    ///
    /// 4. **Apply magnitude:** W_eff = m ⊙ V̂
    ///
    /// 5. **Linear transform:** h = W_eff @ x + bias
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., d_input]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., d_output]
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Step 1: Dequantize base direction
        // Note: Burn's quantization doesn't have fused int8 kernels yet, so this dequantizes fully.
        // Weight is still stored quantized (4-bit), saving memory.
        let v_base_fp = self.v_base.val().dequantize();

        // Step 2: Compute low-rank update ΔV = BA
        let delta_v = self.lora_b.val().matmul(self.lora_a.val());

        // Step 3: Update direction V' = V₀ + ΔV
        // delta_v is [d_output, d_input], transpose to match v_base_fp [d_input, d_output]
        let v_prime = v_base_fp + delta_v.transpose();

        // Step 4: Normalize with detached norms (KEY INNOVATION)
        // This prevents gradients from flowing through the norm computation
        let v_hat = normalize_cols_detached(&v_prime, self.epsilon as f32);

        // Step 5: Apply magnitude W_eff = m ⊙ V̂
        let w_eff = v_hat * self.magnitude.val();

        // Step 6: Linear transform
        linear(input, w_eff, self.bias.as_ref().map(|b| b.val()))
    }
}

impl<B: Backend> ModuleDisplay for QDoRALinear<B> {
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
            .add("quantized", &true)
            .add("bias", &self.bias.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{
        Distribution,
        quantization::{Calibration, QuantLevel, QuantScheme, QuantValue},
    };

    #[test]
    #[ignore] // Requires backend with quantization support (NdArray doesn't support it)
    fn test_qdora_forward() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create and quantize a weight
        let weight_fp = Tensor::<TestBackend, 2>::random([32, 64], Distribution::Default, &device);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor);

        let range =
            burn::tensor::quantization::compute_range(&scheme, &weight_fp, &Calibration::MinMax);
        let qparams = burn::tensor::quantization::compute_q_params(&scheme, range);
        let weight_q = weight_fp.quantize(&scheme, qparams);

        // Create QDoRA layer
        let config = QDoRAConfig::new(32, 64).with_rank(4);
        let layer = config.init_with_quantized_weight(weight_q, None, &device);

        // Forward pass
        let input = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [8, 64]);
    }

    #[test]
    #[ignore] // Requires backend with quantization support (NdArray doesn't support it)
    fn test_qdora_magnitude_initialized() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let weight_fp = Tensor::<TestBackend, 2>::random([16, 32], Distribution::Default, &device);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor);

        let range =
            burn::tensor::quantization::compute_range(&scheme, &weight_fp, &Calibration::MinMax);
        let qparams = burn::tensor::quantization::compute_q_params(&scheme, range);
        let weight_q = weight_fp.quantize(&scheme, qparams);

        let config = QDoRAConfig::new(16, 32).with_rank(4);
        let layer = config.init_with_quantized_weight(weight_q, None, &device);

        // Magnitude should be [1, 32]
        assert_eq!(layer.magnitude.dims(), [1, 32]);

        // All magnitude values should be positive
        let mag_data = layer.magnitude.val().into_data();
        let mag_vec = mag_data.to_vec::<f32>().unwrap();
        for &val in &mag_vec {
            assert!(val > 0.0, "Magnitude should be positive");
        }
    }

    #[test]
    fn display() {
        let device = Default::default();

        let weight_q = Tensor::<TestBackend, 2>::zeros([128, 256], &device);
        let config = QDoRAConfig::new(128, 256).with_rank(16);
        let layer = config.init_with_quantized_weight(weight_q, None, &device);

        let display = alloc::format!("{layer}");
        assert!(display.contains("d_input: 128"));
        assert!(display.contains("d_output: 256"));
        assert!(display.contains("rank: 16"));
        assert!(display.contains("quantized: true"));
    }
}
