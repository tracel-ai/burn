use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend, module::linear};

use crate::ops::merge_lora;

/// Configuration for creating a [`LoRALinear`] layer.
///
/// LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding trainable
/// low-rank matrices to frozen pretrained weights.
///
/// # Mathematical Formulation
///
/// For base weight W₀ ∈ ℝ^{d×k} and input x:
/// - Forward (unmerged): h = W₀x + (α/r) · B(Ax)
/// - Merged (inference): W' = W₀ + (α/r) · BA
///
/// where:
/// - A ∈ ℝ^{r×k}: Adapter matrix (down-projection)
/// - B ∈ ℝ^{d×r}: Adapter matrix (up-projection)
/// - r ≪ min(d,k): Rank of adaptation
/// - α: Scaling hyperparameter (often set to r)
///
/// # Example
///
/// ```rust,ignore
/// use burn_peft::{LoRAConfig, LoRALinear};
///
/// let config = LoRAConfig::new(512, 512)
///     .with_rank(16)
///     .with_alpha(32.0)
///     .with_dropout(0.1);
///
/// let layer = config.init::<MyBackend>(&device);
/// let output = layer.forward(input);
/// ```
#[derive(Config, Debug)]
pub struct LoRAConfig {
    /// Input dimension (d_input)
    pub d_input: usize,
    /// Output dimension (d_output)
    pub d_output: usize,
    /// Rank of the low-rank adaptation (r)
    #[config(default = 8)]
    pub rank: usize,
    /// Scaling factor alpha (α)
    /// Common choice: α = r (so α/r = 1.0)
    #[config(default = 8.0)]
    pub alpha: f64,
    /// Dropout rate applied to input before adapter
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether to include bias in the base linear layer
    #[config(default = true)]
    pub bias: bool,
    /// Initializer for matrix A (down-projection)
    /// Default: Kaiming uniform for stability
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer_a: Initializer,
    /// Initializer for matrix B (up-projection)
    /// Default: Zeros so initial Δ = 0 (no output jump)
    #[config(default = "Initializer::Zeros")]
    pub initializer_b: Initializer,
    /// Initializer for base weight W₀
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer_base: Initializer,
}

impl LoRAConfig {
    /// Initialize a new [`LoRALinear`] module
    pub fn init<B: Backend>(&self, device: &B::Device) -> LoRALinear<B> {
        // Base weight W₀: [d_input, d_output]
        let weight = self.initializer_base.init_with(
            [self.d_input, self.d_output],
            Some(self.d_input),
            Some(self.d_output),
            device,
        );

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
        // Initialized with Kaiming for stable gradients
        let lora_a = self.initializer_a.init_with(
            [self.rank, self.d_input],
            Some(self.rank),
            Some(self.d_input),
            device,
        );

        // LoRA B matrix: [d_output, rank]
        // Initialized to zeros so Δ = B @ A = 0 initially
        let lora_b = self.initializer_b.init_with(
            [self.d_output, self.rank],
            Some(self.d_output),
            Some(self.rank),
            device,
        );

        LoRALinear {
            weight,
            bias,
            lora_a,
            lora_b,
            alpha: self.alpha,
            rank: self.rank,
            dropout: self.dropout,
            merged: false,
        }
    }

    /// Initialize from an existing linear layer's weights
    ///
    /// This allows converting a pretrained linear layer to LoRA by:
    /// 1. Freezing the base weights
    /// 2. Adding trainable low-rank adapters initialized to zero
    ///
    /// # Arguments
    ///
    /// * `weight` - Pretrained weight of shape [d_input, d_output]
    /// * `bias` - Optional pretrained bias of shape [d_output]
    /// * `device` - Device to create new parameters on
    pub fn init_with_base_weight<B: Backend>(
        &self,
        weight: Tensor<B, 2>,
        bias: Option<Tensor<B, 1>>,
        device: &B::Device,
    ) -> LoRALinear<B> {
        // Use pretrained weight (will be frozen during training)
        let weight = Param::from_tensor(weight);

        let bias = bias.map(Param::from_tensor);

        // LoRA adapters initialized same as above
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

        LoRALinear {
            weight,
            bias,
            lora_a,
            lora_b,
            alpha: self.alpha,
            rank: self.rank,
            dropout: self.dropout,
            merged: false,
        }
    }
}

/// LoRA-adapted linear layer
///
/// Implements Low-Rank Adaptation for efficient fine-tuning.
/// Can operate in two modes:
/// 1. Unmerged: h = W₀x + (α/r) · B(Ax) - for training
/// 2. Merged: h = W'x where W' = W₀ + (α/r) · BA - for inference
///
/// # Fields
///
/// - `weight`: Base weight W₀ (typically frozen during fine-tuning)
/// - `lora_a`: Adapter matrix A ∈ ℝ^{r×k}
/// - `lora_b`: Adapter matrix B ∈ ℝ^{d×r}
/// - `alpha`: Scaling factor
/// - `rank`: Rank of adaptation
/// - `dropout`: Dropout rate for regularization
/// - `merged`: Whether adapters are merged into base weight
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LoRALinear<B: Backend> {
    /// Base weight W₀ of shape [d_input, d_output]
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias of shape [d_output]
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// LoRA A matrix of shape [rank, d_input]
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix of shape [d_output, rank]
    pub lora_b: Param<Tensor<B, 2>>,
    /// Scaling factor α
    pub alpha: f64,
    /// Rank r
    pub rank: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Whether adapters are currently merged
    #[module(skip)]
    pub merged: bool,
}

impl<B: Backend> LoRALinear<B> {
    /// Forward pass through LoRA linear layer
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
            // Merged mode: use combined weight
            linear(
                input,
                self.weight.val(),
                self.bias.as_ref().map(|b| b.val()),
            )
        } else {
            // Unmerged mode: base + low-rank path
            // Base output: W₀x
            let base_out = linear(
                input.clone(),
                self.weight.val(),
                self.bias.as_ref().map(|b| b.val()),
            );

            // TODO: Apply dropout to input if enabled
            // For now, we skip dropout application in the tensor path
            // A future version can integrate with burn_nn::Dropout module
            let adapter_input = input;

            // Low-rank path: (α/r) · B(Ax)
            let alpha_over_r = (self.alpha / self.rank as f64) as f32;

            // Compute Ax: input @ A^T
            // A is [rank, d_input], we need [d_input, rank] for linear
            // linear expects weight of shape [d_in, d_out]
            // So we use A.transpose() which gives [d_input, rank]
            let a_out = linear(adapter_input, self.lora_a.val().transpose(), None);
            // a_out shape: [..., rank]

            // Compute B(Ax): a_out @ B^T
            // B is [d_output, rank], we need [rank, d_output] for linear
            let delta = linear(a_out, self.lora_b.val().transpose(), None);
            // delta shape: [..., d_output]

            // Scale and combine: W₀x + (α/r) · B(Ax)
            base_out + delta * alpha_over_r
        }
    }

    /// Merge LoRA adapters into base weight for inference
    ///
    /// Computes W' = W₀ + (α/r) · BA and updates the base weight.
    /// After merging, forward passes use only the merged weight (faster).
    ///
    /// This is useful for deployment where you want the efficiency of
    /// a single matrix multiplication without the adapter overhead.
    pub fn merge_weights(&mut self) {
        if !self.merged {
            let alpha_over_r = (self.alpha / self.rank as f64) as f32;

            // Compute merged weight: W₀ + (α/r) · BA
            let merged_weight = merge_lora(
                &self.weight.val(),
                &self.lora_b.val(),
                &self.lora_a.val(),
                alpha_over_r,
            );

            // Update base weight
            self.weight = Param::from_tensor(merged_weight);
            self.merged = true;
        }
    }

    /// Unmerge LoRA adapters from base weight
    ///
    /// Computes W₀ = W' - (α/r) · BA to restore the original base weight.
    /// After unmerging, forward passes compute adapters separately.
    pub fn unmerge_weights(&mut self) {
        if self.merged {
            let alpha_over_r = (self.alpha / self.rank as f64) as f32;

            // Compute original base weight: W' - (α/r) · BA
            let base_weight = crate::ops::unmerge_lora(
                &self.weight.val(),
                &self.lora_b.val(),
                &self.lora_a.val(),
                alpha_over_r,
            );

            // Restore base weight
            self.weight = Param::from_tensor(base_weight);
            self.merged = false;
        }
    }

    /// Check if adapters are currently merged
    pub fn is_merged(&self) -> bool {
        self.merged
    }
}

impl<B: Backend> ModuleDisplay for LoRALinear<B> {
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
            .add("bias", &self.bias.is_some())
            .add("merged", &self.merged)
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Distribution, Tensor, TensorData};

    #[test]
    fn test_lora_initialization() {
        let device = Default::default();

        let config = LoRAConfig::new(64, 128).with_rank(8).with_alpha(16.0);

        let layer = config.init::<TestBackend>(&device);

        // Check shapes
        assert_eq!(layer.weight.dims(), [64, 128]); // [d_input, d_output]
        assert_eq!(layer.lora_a.dims(), [8, 64]); // [rank, d_input]
        assert_eq!(layer.lora_b.dims(), [128, 8]); // [d_output, rank]

        // Check B is initialized to zeros (so Δ = 0)
        let b_data = layer.lora_b.val().into_data();
        let b_vec = b_data.to_vec::<f32>().unwrap();
        for &val in &b_vec {
            assert!((val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lora_forward_unmerged() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let config = LoRAConfig::new(32, 64)
            .with_rank(4)
            .with_alpha(4.0)
            .with_bias(false);

        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random([8, 32], Distribution::Default, &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [8, 64]);
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        let config = LoRAConfig::new(16, 32).with_rank(4).with_alpha(8.0);

        let mut layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::random([4, 16], Distribution::Default, &device);

        // Get output in unmerged mode
        let output_unmerged = layer.forward(input.clone());

        // Merge weights
        layer.merge_weights();
        assert!(layer.is_merged());

        // Get output in merged mode
        let output_merged = layer.forward(input.clone());

        // Outputs should be identical (within numerical precision)
        let diff = (output_unmerged.clone() - output_merged.clone())
            .abs()
            .mean();
        assert!(diff.into_scalar() < 1e-5);

        // Unmerge weights
        layer.unmerge_weights();
        assert!(!layer.is_merged());

        // Output should match original unmerged output
        let output_restored = layer.forward(input);
        let diff = (output_unmerged - output_restored).abs().mean();
        assert!(diff.into_scalar() < 1e-5);
    }

    #[test]
    fn test_lora_zero_init_no_jump() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Initialize with constant base weight for predictability
        let config = LoRAConfig::new(8, 16)
            .with_rank(4)
            .with_bias(false)
            .with_initializer_base(Initializer::Constant { value: 1.0 });

        let layer = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones([2, 8], &device);

        // Base output (W₀ @ x) where W₀ is all 1s
        // Each output element should be sum of 8 ones = 8
        let base_expected =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[8.0; 16], [8.0; 16]]), &device);

        // LoRA output should equal base output since B=0 initially
        let output = layer.forward(input);

        let diff = (output - base_expected).abs().mean();
        assert!(diff.into_scalar() < 1e-5);
    }

    #[test]
    fn display() {
        let device = Default::default();
        let config = LoRAConfig::new(128, 256).with_rank(16).with_alpha(32.0);
        let layer = config.init::<TestBackend>(&device);

        let display = alloc::format!("{layer}");
        assert!(display.contains("d_input: 128"));
        assert!(display.contains("d_output: 256"));
        assert!(display.contains("rank: 16"));
        assert!(display.contains("alpha: 32"));
    }
}
