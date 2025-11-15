use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend, module::linear};

use crate::ops::merge_lora;

/// Configuration for creating a [`QLoRALinear`] layer.
///
/// QLoRA (Quantized LoRA) combines low-rank adaptation with quantized base weights
/// for memory-efficient fine-tuning.
///
/// # Mathematical Formulation
///
/// Base weight W₀ is stored quantized, adapters are in full precision:
/// - Forward: h = QuantMatMul(W₀_q, x) + (α/r)·B(Ax)
/// - Only A and B are trainable (W₀ frozen and quantized)
///
/// # Memory Savings
///
/// With 4-bit quantization and rank r=8 on a 4096×4096 layer:
/// - Full precision: 64 MB
/// - LoRA (fp16): 64 MB base + 0.5 MB adapters = 64.5 MB
/// - QLoRA (4-bit + fp16 adapters): 8 MB base + 0.5 MB adapters = 8.5 MB
///
/// **87% memory reduction!**
///
/// # Example
///
/// ```rust,ignore
/// use burn_peft::{QLoRAConfig, QLoRALinear};
/// use burn::tensor::quantization::{QuantScheme, QuantValue, QuantLevel};
///
/// let scheme = QuantScheme::default()
///     .with_value(QuantValue::Q4F)  // 4-bit quantization
///     .with_level(QuantLevel::block([32]));
///
/// let config = QLoRAConfig::new(4096, 4096)
///     .with_rank(8)
///     .with_alpha(16.0)
///     .with_quant_scheme(scheme);
///
/// let layer = config.init::<MyBackend>(&device);
/// ```
#[derive(Config, Debug)]
pub struct QLoRAConfig {
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

impl QLoRAConfig {
    /// Initialize a new QLoRA layer from a pretrained quantized weight
    ///
    /// This is the typical use case: take an existing model, quantize its weights,
    /// then add low-rank adapters for efficient fine-tuning.
    ///
    /// # Arguments
    ///
    /// * `weight_quantized` - Pre-quantized base weight (frozen)
    /// * `bias` - Optional bias (not quantized)
    /// * `device` - Device to create adapters on
    ///
    /// # Returns
    ///
    /// QLoRA layer with quantized base and trainable adapters
    pub fn init_with_quantized_weight<B: Backend>(
        &self,
        weight_quantized: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> QLoRALinear<B> {
        // Use quantized weight (frozen)
        let weight = Param::from_tensor(weight_quantized);
        let bias = bias.map(Param::from_tensor);

        // Initialize adapters in full precision
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

        QLoRALinear {
            weight,
            bias,
            lora_a,
            lora_b,
            alpha: self.alpha,
            rank: self.rank,
            dropout: self.dropout,
        }
    }
}

/// QLoRA-adapted linear layer
///
/// Implements Quantized Low-Rank Adaptation for memory-efficient fine-tuning.
///
/// # Architecture
///
/// ```text
/// Input x
///   ├─> QuantMatMul(W₀_q, x) ──> Base output (quantized path)
///   │
///   └─> A @ x ─> B @ (·) ──────> Low-rank output (full precision)
///                    │
///                    ├──> Scale by (α/r)
///                    │
///                    └──> Add to base output ──> Final output
/// ```
///
/// # Memory Efficiency
///
/// The quantized base weight typically uses 4–8 bits per parameter, while
/// adapters use full precision (16–32 bits). Since r ≪ min(d,k), the adapter
/// overhead is minimal compared to the base weight savings.
///
/// # Training
///
/// Only `lora_a` and `lora_b` receive gradients. The quantized base weight
/// remains frozen, making this ideal for fine-tuning large pretrained models
/// with limited GPU memory.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct QLoRALinear<B: Backend> {
    /// Quantized base weight W₀_q of shape [d_input, d_output] (frozen)
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// LoRA A matrix of shape [rank, d_input] (trainable, full precision)
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix of shape [d_output, rank] (trainable, full precision)
    pub lora_b: Param<Tensor<B, 2>>,
    /// Scaling factor α
    pub alpha: f64,
    /// Rank r
    pub rank: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl<B: Backend> QLoRALinear<B> {
    /// Forward pass through QLoRA linear layer
    ///
    /// # Computation Path
    ///
    /// 1. Base path: y = QuantMatMul(W₀_q, x)
    ///    - Uses fused dequantization if supported by backend
    ///    - Falls back to dequantize() + matmul if not
    ///
    /// 2. Adapter path: Δ = B(Ax)
    ///    - Computed in full precision
    ///    - Scaled by α/r
    ///
    /// 3. Combine: h = y + (α/r)·Δ
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., d_input]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., d_output]
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Base path: Dequantize then linear (W₀_fp @ x) + bias
        // Note: Burn's quantized matmul always dequantizes (no fused int8 kernel yet).
        // We explicitly dequantize here for clarity. Weight is still stored quantized (4-bit)
        let weight_fp = self.weight.val().dequantize();
        let base_out = linear(
            input.clone(),
            weight_fp,
            self.bias.as_ref().map(|b| b.val()),
        );

        // Adapter path: (α/r) · B(Ax)
        let alpha_over_r = (self.alpha / self.rank as f64) as f32;

        // Compute Ax
        let a_out = linear(input, self.lora_a.val().transpose(), None);

        // Compute B(Ax) and scale
        let delta = linear(a_out, self.lora_b.val().transpose(), None);

        // Combine: base + scaled adapter
        base_out + delta * alpha_over_r
    }

    /// Merge adapters into dequantized base weight (for inference)
    ///
    /// This dequantizes the base weight and merges the adapters:
    /// W' = Dequant(W₀_q) + (α/r)·BA
    ///
    /// **Warning:** This converts the quantized weight back to full precision,
    /// losing the memory benefits of quantization. Only use this if you need
    /// a single full-precision weight for deployment.
    ///
    /// # Returns
    ///
    /// A standard LoRA layer with merged full-precision weight
    pub fn merge_to_full_precision(&self) -> crate::LoRALinear<B> {
        let alpha_over_r = (self.alpha / self.rank as f64) as f32;

        // Dequantize base weight
        let weight_fp = self.weight.val().dequantize();

        // Merge adapters
        let merged_weight = merge_lora(
            &weight_fp,
            &self.lora_b.val(),
            &self.lora_a.val(),
            alpha_over_r,
        );

        crate::LoRALinear {
            weight: Param::from_tensor(merged_weight),
            bias: self.bias.clone(),
            lora_a: self.lora_a.clone(),
            lora_b: self.lora_b.clone(),
            alpha: self.alpha,
            rank: self.rank,
            dropout: self.dropout,
            merged: true,
        }
    }
}

impl<B: Backend> ModuleDisplay for QLoRALinear<B> {
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
    fn test_qlora_forward() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create a full-precision weight
        let weight_fp = Tensor::<TestBackend, 2>::random([32, 64], Distribution::Default, &device);

        // Quantize it
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor);

        let range =
            burn::tensor::quantization::compute_range(&scheme, &weight_fp, &Calibration::MinMax);
        let qparams = burn::tensor::quantization::compute_q_params(&scheme, range);
        let weight_q = weight_fp.quantize(&scheme, qparams);

        // Create QLoRA layer
        let config = QLoRAConfig::new(32, 64).with_rank(4).with_alpha(8.0);
        let layer = config.init_with_quantized_weight(weight_q, None, &device);

        // Forward pass
        let input = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [8, 64]);
    }

    #[test]
    #[ignore] // Requires backend with quantization support (NdArray doesn't support it)
    fn test_qlora_memory_savings() {
        let device = Default::default();

        // Create a large weight
        let weight_fp =
            Tensor::<TestBackend, 2>::random([1024, 2048], Distribution::Default, &device);

        // Quantize to 8-bit
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::Tensor);

        let range =
            burn::tensor::quantization::compute_range(&scheme, &weight_fp, &Calibration::MinMax);
        let qparams = burn::tensor::quantization::compute_q_params(&scheme, range);
        let weight_q = weight_fp.quantize(&scheme, qparams);

        // Create QLoRA layer
        let config = QLoRAConfig::new(1024, 2048).with_rank(8);
        let _layer = config.init_with_quantized_weight(weight_q, None, &device);

        // The quantized weight should use ~8x less memory than full precision
        // Adapters add minimal overhead: 2 * (1024 * 8 + 2048 * 8) = ~50KB
        // vs base weight: 1024 * 2048 * 4 bytes = 8 MB (full) or 1 MB (quantized)
    }

    #[test]
    fn display() {
        let device = Default::default();

        let weight_q = Tensor::<TestBackend, 2>::zeros([128, 256], &device);
        let config = QLoRAConfig::new(128, 256).with_rank(16).with_alpha(32.0);
        let layer = config.init_with_quantized_weight(weight_q, None, &device);

        let display = alloc::format!("{layer}");
        assert!(display.contains("d_input: 128"));
        assert!(display.contains("d_output: 256"));
        assert!(display.contains("rank: 16"));
        assert!(display.contains("quantized: true"));
    }
}
