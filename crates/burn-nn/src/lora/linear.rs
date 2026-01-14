use alloc::format;

use burn_core as burn;

use burn::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param};
use burn::tensor::{Tensor, backend::Backend};

use crate::modules::{Dropout, DropoutConfig, Linear};

use super::LoraAdaptable;
use super::adapter::LoraLinearAdapter;
use super::config::{LoraBias, LoraConfig};

#[cfg(feature = "std")]
use burn::record::{FileRecorder, RecorderError};
#[cfg(feature = "std")]
use std::path::PathBuf;

/// Error type for LoRA adapter operations.
#[derive(Debug, Clone)]
pub enum LoraError {
    /// Adapter dimensions don't match layer dimensions.
    DimensionMismatch {
        /// Expected input dimension from the layer.
        expected_d_input: usize,
        /// Expected output dimension from the layer.
        expected_d_output: usize,
        /// Actual input dimension from the adapter.
        actual_d_input: usize,
        /// Actual output dimension from the adapter.
        actual_d_output: usize,
    },
}

impl core::fmt::Display for LoraError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LoraError::DimensionMismatch {
                expected_d_input,
                expected_d_output,
                actual_d_input,
                actual_d_output,
            } => {
                write!(
                    f,
                    "Adapter dimensions [{}, {}] don't match layer dimensions [{}, {}]",
                    actual_d_input, actual_d_output, expected_d_input, expected_d_output
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoraError {}

/// Linear layer with LoRA (Low-Rank Adaptation) applied.
///
/// This wrapper adds trainable low-rank matrices A and B to a frozen base
/// linear layer. The forward pass computes:
///
/// ```text
/// output = base(x) + dropout(x) @ A @ B * scaling
/// ```
///
/// # Fields
///
/// - `base`: The frozen base linear layer
/// - `lora_a`: Down-projection matrix `[d_input, rank]`
/// - `lora_b`: Up-projection matrix `[rank, d_output]`
/// - `scaling`: Precomputed scaling factor (`alpha / rank`)
/// - `dropout`: Dropout applied to LoRA branch input
///
/// # Example
///
/// ```ignore
/// use burn_nn::{Linear, LinearConfig};
/// use burn_nn::lora::{LoraConfig, LoraAdaptable};
///
/// let linear = LinearConfig::new(768, 768).init(&device);
/// let lora_linear = linear.with_lora(&LoraConfig::new(16), &device);
///
/// // Forward pass
/// let output = lora_linear.forward(input);
///
/// // Merge for inference
/// let merged = lora_linear.merge();
/// ```
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LoraLinear<B: Backend> {
    /// Base linear layer (frozen).
    pub base: Linear<B>,
    /// LoRA A matrix (down-projection): `[d_input, rank]`.
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA B matrix (up-projection): `[rank, d_output]`.
    pub lora_b: Param<Tensor<B, 2>>,
    /// Precomputed scaling factor: `alpha / rank` (or `alpha / sqrt(rank)` for RSLoRA).
    pub scaling: f64,
    /// Dropout applied to LoRA branch input (only active during training).
    pub dropout: Dropout,
    /// Original LoRA configuration (preserved for adapter extraction).
    /// Wrapped in `Ignored` so it doesn't require Module implementation.
    /// Config is fully preserved when using `into_adapter()` / adapter persistence.
    pub config: Ignored<LoraConfig>,
}

impl<B: Backend> LoraAdaptable<B> for Linear<B> {
    type Wrapped = LoraLinear<B>;

    fn lora_dims(&self) -> (usize, usize) {
        let [d_in, d_out] = self.weight.shape().dims::<2>();
        (d_in, d_out)
    }

    fn with_lora(self, config: &LoraConfig, device: &B::Device) -> LoraLinear<B> {
        config.validate();
        let (d_in, d_out) = self.lora_dims();

        // Freeze base weights, optionally unfreeze bias
        let base = match config.bias {
            LoraBias::None => self.no_grad(),
            LoraBias::All => {
                let mut frozen = self.no_grad();
                frozen.bias = frozen.bias.map(|b| b.set_require_grad(true));
                frozen
            }
        };

        LoraLinear {
            base,
            lora_a: config.init_a(d_in, device),
            lora_b: config.init_b(d_out, device),
            scaling: config.scaling(),
            dropout: DropoutConfig::new(config.dropout).init(),
            config: Ignored(config.clone()),
        }
    }
}

impl<B: Backend> LoraLinear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// Computes: `base(x) + dropout(x) @ A @ B * scaling`
    ///
    /// # Arguments
    ///
    /// - `input` - The input tensor of shape `[..., d_input]`
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    ///
    /// # Returns
    ///
    /// The transformed tensor of shape `[..., d_output]`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Handle 1D input by temporarily upgrading to 2D
        // (unsqueeze::<D>() on 2D LoRA weights fails when D=1)
        if D == 1 {
            let input_2d: Tensor<B, 2> = input.unsqueeze_dim(0);
            let output_2d = self.forward_nd(input_2d);
            return output_2d.squeeze_dim(0);
        }
        self.forward_nd(input)
    }

    /// Internal forward pass for tensors with D >= 2.
    fn forward_nd<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // LoRA branch: dropout(x) @ A @ B * scaling
        // Clone is required since both base and LoRA paths need the input.
        // We compute LoRA first so dropout consumes the clone.
        let lora_in = self.dropout.forward(input.clone());
        let lora_a = self.lora_a.val().unsqueeze::<D>();
        let lora_b = self.lora_b.val().unsqueeze::<D>();
        let lora_out = lora_in
            .matmul(lora_a)
            .matmul(lora_b)
            .mul_scalar(self.scaling);

        // Base path consumes original input
        self.base.forward(input) + lora_out
    }

    /// Merge LoRA weights into the base layer.
    ///
    /// Computes: `W_merged = W_base + A @ B * scaling`
    ///
    /// This eliminates all LoRA overhead at inference time. The returned
    /// `Linear` layer produces identical outputs to the `LoraLinear` layer
    /// (ignoring dropout which is disabled during inference).
    ///
    /// The merge preserves the `ParamId` and `ParamMapper` from the base
    /// layer, ensuring checkpoint compatibility.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Train with LoRA
    /// let lora_linear = linear.with_lora(&config, &device);
    /// // ... training ...
    ///
    /// // Merge for inference
    /// let inference_linear = lora_linear.merge();
    /// ```
    pub fn merge(self) -> Linear<B> {
        // Preserve Param infrastructure (ID, mapper for LinearLayout::Col)
        let (id, weight, mapper) = self.base.weight.consume();

        // Compute merged weight: W + A @ B * scaling
        let delta = self
            .lora_a
            .val()
            .matmul(self.lora_b.val())
            .mul_scalar(self.scaling);
        let merged = (weight + delta).set_require_grad(false);

        Linear {
            weight: Param::from_mapped_value(id, merged, mapper),
            bias: self.base.bias,
        }
    }

    /// Returns the LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.shape().dims::<2>()[1]
    }

    /// Returns the input dimension of the base layer.
    pub fn d_input(&self) -> usize {
        self.base.weight.shape().dims::<2>()[0]
    }

    /// Returns the output dimension of the base layer.
    pub fn d_output(&self) -> usize {
        self.base.weight.shape().dims::<2>()[1]
    }

    /// Extract adapter weights from this LoRA layer.
    ///
    /// Returns a `LoraLinearAdapter` containing only the LoRA matrices (A and B)
    /// and full configuration. This does NOT include base model weights.
    ///
    /// Use this to save trained LoRA adapters independently of the base model,
    /// enabling adapter swapping and efficient storage.
    ///
    /// The full `LoraConfig` is preserved, including `rank`, `alpha`, `dropout`,
    /// `bias`, `use_rslora`, and `init` settings. This allows training to be
    /// resumed with the exact same configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let adapter = lora_linear.clone().into_adapter();
    /// // adapter contains lora_a, lora_b, and full config
    /// ```
    pub fn into_adapter(self) -> LoraLinearAdapter<B> {
        LoraLinearAdapter::new(self.lora_a, self.lora_b, self.config.0)
    }

    /// Load adapter weights into this LoRA layer.
    ///
    /// Replaces the current LoRA matrices (A and B) with those from the adapter.
    /// The base layer weights remain unchanged. The full config is also restored,
    /// enabling training continuation with the same settings.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter containing new LoRA weights and config
    ///
    /// # Returns
    ///
    /// `Ok(Self)` with updated LoRA weights, scaling, and config, or
    /// `Err(LoraError::DimensionMismatch)` if adapter dimensions don't match.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Swap to a different adapter at runtime
    /// lora_linear = lora_linear.load_adapter(new_adapter)?;
    /// ```
    pub fn load_adapter(mut self, adapter: LoraLinearAdapter<B>) -> Result<Self, LoraError> {
        let expected_d_input = self.d_input();
        let expected_d_output = self.d_output();
        let actual_d_input = adapter.d_input();
        let actual_d_output = adapter.d_output();

        if expected_d_input != actual_d_input || expected_d_output != actual_d_output {
            return Err(LoraError::DimensionMismatch {
                expected_d_input,
                expected_d_output,
                actual_d_input,
                actual_d_output,
            });
        }

        self.lora_a = adapter.lora_a;
        self.lora_b = adapter.lora_b;
        self.scaling = adapter.config.scaling();
        self.config = Ignored(adapter.config);
        Ok(self)
    }

    /// Save adapter weights to a file.
    ///
    /// Saves only the LoRA matrices (A and B) and configuration to a MessagePack file.
    /// The base model weights are NOT saved - only the small adapter weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the adapter (file extension added automatically)
    /// * `recorder` - The recorder to use (e.g., `NamedMpkFileRecorder`)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `RecorderError` on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
    ///
    /// let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    /// lora_linear.save_adapter("./my-adapter", &recorder)?;
    /// // Creates ./my-adapter.mpk
    /// ```
    #[cfg(feature = "std")]
    pub fn save_adapter<R>(
        &self,
        path: impl Into<PathBuf>,
        recorder: &R,
    ) -> Result<(), RecorderError>
    where
        R: FileRecorder<B>,
    {
        let adapter = self.clone().into_adapter();
        recorder.record(adapter, path.into())
    }

    /// Load adapter weights from a file.
    ///
    /// Loads LoRA matrices and configuration from a MessagePack file and applies
    /// them to this layer. The base model weights remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the adapter file (file extension added automatically)
    /// * `recorder` - The recorder to use (must match the one used for saving)
    /// * `device` - Device to load the tensors onto
    ///
    /// # Returns
    ///
    /// Self with loaded adapter weights, or `RecorderError` on failure
    /// (including dimension mismatch errors).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
    ///
    /// let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    /// lora_linear = lora_linear.load_adapter_file("./my-adapter", &recorder, &device)?;
    /// ```
    #[cfg(feature = "std")]
    pub fn load_adapter_file<R>(
        self,
        path: impl Into<PathBuf>,
        recorder: &R,
        device: &B::Device,
    ) -> Result<Self, RecorderError>
    where
        R: FileRecorder<B>,
    {
        let adapter: LoraLinearAdapter<B> = recorder.load(path.into(), device)?;
        self.load_adapter(adapter)
            .map_err(|e| RecorderError::Unknown(format!("{e}")))
    }
}

impl<B: Backend> ModuleDisplay for LoraLinear<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.base.weight.shape().dims::<2>();
        let rank = self.rank();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("rank", &rank)
            .add("bias", &self.base.bias.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::lora::config::LoraInit;
    use crate::modules::LinearConfig;
    use burn::tensor::ops::FloatElem;
    use burn::tensor::{Distribution, Shape, Tolerance};

    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_lora_linear_identity_start() {
        // With B=zeros (default Kaiming init), output should match base
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let base_weight = linear.weight.val().clone();

        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let base_out = burn::tensor::module::linear(
            input.clone(),
            base_weight,
            lora.base.bias.as_ref().map(|b| b.val()),
        );
        let lora_out = lora.forward(input);

        base_out
            .into_data()
            .assert_approx_eq::<FT>(&lora_out.into_data(), Tolerance::default());
    }

    #[test]
    fn test_lora_linear_merge_equivalence() {
        // After merge, output should match LoRA output (with non-zero LoRA weights)
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let lora_out = lora.forward(input.clone());

        let merged = lora.merge();
        let merged_out = merged.forward(input);

        lora_out
            .into_data()
            .assert_approx_eq::<FT>(&merged_out.into_data(), Tolerance::default());
    }

    #[test]
    fn test_lora_linear_dimensions() {
        let device = Default::default();

        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        assert_eq!(lora.d_input(), 64);
        assert_eq!(lora.d_output(), 32);
        assert_eq!(lora.rank(), 8);
        assert_eq!(lora.lora_a.shape().dims::<2>(), [64, 8]);
        assert_eq!(lora.lora_b.shape().dims::<2>(), [8, 32]);
    }

    #[test]
    fn test_lora_linear_bias_none() {
        let device = Default::default();

        let linear = LinearConfig::new(64, 32)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_bias(LoraBias::None), &device);

        // Base weight should be frozen
        assert!(!lora.base.weight.is_require_grad());
        // Bias should also be frozen with LoraBias::None
        assert!(lora.base.bias.is_some());
        assert!(!lora.base.bias.as_ref().unwrap().is_require_grad());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_lora_linear_bias_all() {
        use crate::TestAutodiffBackend;

        let device = Default::default();

        let linear = LinearConfig::new(64, 32)
            .with_bias(true)
            .init::<TestAutodiffBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_bias(LoraBias::All), &device);

        // Base weight should be frozen
        assert!(!lora.base.weight.is_require_grad());
        // Bias should be trainable with LoraBias::All
        assert!(lora.base.bias.is_some());
        assert!(lora.base.bias.as_ref().unwrap().is_require_grad());
    }

    #[test]
    fn test_lora_linear_scaling() {
        let device = Default::default();

        // Test standard scaling: alpha / rank
        let config = LoraConfig::new(8).with_alpha(16.0);
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&config, &device);
        assert!((lora.scaling - 2.0).abs() < 1e-10);

        // Test RSLoRA scaling: alpha / sqrt(rank)
        let config = LoraConfig::new(16).with_alpha(16.0).with_use_rslora(true);
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&config, &device);
        assert!((lora.scaling - 4.0).abs() < 1e-10); // 16 / sqrt(16) = 4
    }

    #[test]
    fn test_lora_linear_forward_shapes() {
        let device = Default::default();

        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        // Test 2D input
        let input_2d = Tensor::<TestBackend, 2>::random([4, 64], Distribution::Default, &device);
        let output_2d = lora.forward(input_2d);
        assert_eq!(output_2d.shape(), Shape::new([4, 32]));

        // Test 3D input
        let input_3d = Tensor::<TestBackend, 3>::random([2, 4, 64], Distribution::Default, &device);
        let output_3d = lora.forward(input_3d);
        assert_eq!(output_3d.shape(), Shape::new([2, 4, 32]));
    }

    #[test]
    fn test_lora_linear_display() {
        let device = Default::default();

        let linear = LinearConfig::new(64, 32)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        let display = alloc::format!("{lora}");
        assert!(display.contains("d_input: 64"));
        assert!(display.contains("d_output: 32"));
        assert!(display.contains("rank: 8"));
        assert!(display.contains("bias: true"));
    }

    #[test]
    fn test_lora_linear_no_bias() {
        let device = Default::default();

        let linear = LinearConfig::new(64, 32)
            .with_bias(false)
            .init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        assert!(lora.base.bias.is_none());

        // Should still work
        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let _output = lora.forward(input);
    }

    #[test]
    fn test_into_adapter_preserves_weights_and_full_config() {
        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create LoRA with non-default config values
        let config = LoraConfig::new(8)
            .with_alpha(16.0)
            .with_dropout(0.1)
            .with_bias(LoraBias::All)
            .with_use_rslora(true)
            .with_init(LoraInit::Gaussian);

        let linear = LinearConfig::new(64, 32)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let lora = linear.with_lora(&config, &device);

        // Get original weights
        let original_a = lora.lora_a.val().clone();
        let original_b = lora.lora_b.val().clone();

        // Extract adapter
        let adapter = lora.into_adapter();

        // Verify adapter has same weights
        original_a
            .into_data()
            .assert_approx_eq::<FT>(&adapter.lora_a.val().into_data(), Tolerance::default());
        original_b
            .into_data()
            .assert_approx_eq::<FT>(&adapter.lora_b.val().into_data(), Tolerance::default());

        // Verify FULL config is preserved (not just rank and alpha)
        assert_eq!(adapter.config.rank, 8);
        assert!((adapter.config.alpha - 16.0).abs() < 1e-10);
        assert!((adapter.config.dropout - 0.1).abs() < 1e-10);
        assert_eq!(adapter.config.bias, LoraBias::All);
        assert!(adapter.config.use_rslora);
        assert_eq!(adapter.config.init, LoraInit::Gaussian);
    }

    #[test]
    fn test_load_adapter_updates_weights() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        // Create two LoRA layers with different weights
        let linear1 = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora1 = linear1.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        TestBackend::seed(&device, 123); // Different seed
        let linear2 = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora2 = linear2.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        // Extract adapter from lora2
        let adapter2 = lora2.clone().into_adapter();
        let adapter2_a = adapter2.lora_a.val().clone();
        let adapter2_b = adapter2.lora_b.val().clone();

        // Load adapter2 into lora1
        let lora1_updated = lora1.load_adapter(adapter2).unwrap();

        // Verify lora1 now has lora2's weights
        lora1_updated
            .lora_a
            .val()
            .into_data()
            .assert_approx_eq::<FT>(&adapter2_a.into_data(), Tolerance::default());
        lora1_updated
            .lora_b
            .val()
            .into_data()
            .assert_approx_eq::<FT>(&adapter2_b.into_data(), Tolerance::default());
    }

    #[test]
    fn test_adapter_output_equivalence() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        // Create LoRA layer
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        // Get output before extraction
        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let output_before = lora.forward(input.clone());

        // Extract adapter and load onto fresh layer with same base
        let adapter = lora.clone().into_adapter();

        // Reload the adapter onto a clone (base weights preserved)
        let lora_reloaded = lora.clone().load_adapter(adapter).unwrap();

        // Get output after reload
        let output_after = lora_reloaded.forward(input);

        // Should produce identical output
        output_before
            .into_data()
            .assert_approx_eq::<FT>(&output_after.into_data(), Tolerance::default());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_adapter_save_load_roundtrip() {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        use std::fs;

        let device = Default::default();
        TestBackend::seed(&device, 42);

        // Create and configure LoRA layer with Gaussian init (non-zero)
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(
            &LoraConfig::new(8)
                .with_alpha(16.0)
                .with_init(LoraInit::Gaussian),
            &device,
        );

        // Get original LoRA weights for comparison
        let original_a = lora.lora_a.val().clone();
        let original_b = lora.lora_b.val().clone();

        // Save adapter to temp file
        let temp_dir = std::env::temp_dir();
        let adapter_path = temp_dir.join("test_lora_adapter_roundtrip");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        lora.save_adapter(&adapter_path, &recorder)
            .expect("Failed to save adapter");

        // Create fresh LoRA layer with same base (clone the original)
        // and zero-initialized LoRA weights
        let fresh_lora = lora.clone();

        // Load adapter - this should restore the original LoRA weights
        let loaded_lora = fresh_lora
            .load_adapter_file(&adapter_path, &recorder, &device)
            .expect("Failed to load adapter");

        // Verify LoRA weights are identical after roundtrip
        loaded_lora
            .lora_a
            .val()
            .into_data()
            .assert_approx_eq::<FT>(&original_a.into_data(), Tolerance::default());
        loaded_lora
            .lora_b
            .val()
            .into_data()
            .assert_approx_eq::<FT>(&original_b.into_data(), Tolerance::default());

        // Verify outputs are identical
        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let output_original = lora.forward(input.clone());
        let output_loaded = loaded_lora.forward(input);

        output_original
            .into_data()
            .assert_approx_eq::<FT>(&output_loaded.into_data(), Tolerance::default());

        // Cleanup
        let _ = fs::remove_file(adapter_path.with_extension("mpk"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_adapter_swap_at_runtime() {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        use std::fs;

        let device = Default::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let temp_dir = std::env::temp_dir();

        // Create two different adapters
        TestBackend::seed(&device, 1);
        let linear1 = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora1 = linear1.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);
        let adapter1_path = temp_dir.join("test_adapter1");
        lora1.save_adapter(&adapter1_path, &recorder).unwrap();

        TestBackend::seed(&device, 2);
        let linear2 = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora2 = linear2.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);
        let adapter2_path = temp_dir.join("test_adapter2");
        lora2.save_adapter(&adapter2_path, &recorder).unwrap();

        // Create base LoRA layer
        TestBackend::seed(&device, 0);
        let base_linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let mut lora = base_linear.with_lora(&LoraConfig::new(8), &device);

        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);

        // Load adapter1 and get output
        lora = lora
            .load_adapter_file(&adapter1_path, &recorder, &device)
            .unwrap();
        let output1 = lora.forward(input.clone());

        // Swap to adapter2 and get output
        lora = lora
            .load_adapter_file(&adapter2_path, &recorder, &device)
            .unwrap();
        let output2 = lora.forward(input.clone());

        // Outputs should be different (different adapters)
        use burn::tensor::ElementConversion;
        let diff = (output1.clone() - output2.clone()).abs().sum();
        assert!(
            diff.into_scalar().elem::<f32>() > 0.01,
            "Outputs should differ with different adapters"
        );

        // Swap back to adapter1 and verify we get the original output
        lora = lora
            .load_adapter_file(&adapter1_path, &recorder, &device)
            .unwrap();
        let output1_again = lora.forward(input);

        output1
            .into_data()
            .assert_approx_eq::<FT>(&output1_again.into_data(), Tolerance::default());

        // Cleanup
        let _ = fs::remove_file(adapter1_path.with_extension("mpk"));
        let _ = fs::remove_file(adapter2_path.with_extension("mpk"));
    }

    #[test]
    fn test_load_adapter_dimension_mismatch() {
        let device = Default::default();

        // Create LoRA layers with different dimensions
        let linear_64_32 = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora_64_32 = linear_64_32.with_lora(&LoraConfig::new(8), &device);

        let linear_128_64 = LinearConfig::new(128, 64).init::<TestBackend>(&device);
        let lora_128_64 = linear_128_64.with_lora(&LoraConfig::new(8), &device);

        // Extract adapter from mismatched layer
        let adapter_128_64 = lora_128_64.into_adapter();

        // Should fail with dimension mismatch
        let result = lora_64_32.load_adapter(adapter_128_64);
        assert!(result.is_err());

        // Verify error message contains expected dimensions
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("128"));
        assert!(msg.contains("64"));
    }

    /// Test that gradients only flow to LoRA parameters, not to frozen base weights.
    #[cfg(feature = "std")]
    #[test]
    fn test_gradient_flow_only_lora_params() {
        use crate::TestAutodiffBackend;

        let device = Default::default();
        let linear = LinearConfig::new(64, 32).init::<TestAutodiffBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        let input =
            Tensor::<TestAutodiffBackend, 2>::random([2, 64], Distribution::Default, &device);
        let output = lora.forward(input);
        let loss = output.sum();
        let grads = loss.backward();

        // LoRA params should have gradients
        assert!(
            lora.lora_a.grad(&grads).is_some(),
            "LoRA A should have gradients"
        );
        assert!(
            lora.lora_b.grad(&grads).is_some(),
            "LoRA B should have gradients"
        );

        // Base weight should NOT have gradients (frozen via no_grad)
        assert!(
            lora.base.weight.grad(&grads).is_none(),
            "Base weight should not have gradients (frozen)"
        );
    }

    /// Test that alpha=0 results in no LoRA contribution.
    #[test]
    fn test_lora_alpha_zero_scaling() {
        let device = Default::default();
        let config = LoraConfig::new(8).with_alpha(0.0);

        // Scaling should be zero
        assert!(
            (config.scaling() - 0.0).abs() < 1e-10,
            "Scaling should be zero when alpha is zero"
        );

        // With Gaussian init (non-zero LoRA weights), output should still equal base
        // because the scaling factor is 0
        TestBackend::seed(&device, 42);
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let base_weight = linear.weight.val().clone();
        let base_bias = linear.bias.as_ref().map(|b| b.val().clone());

        let lora = linear.with_lora(&config.with_init(LoraInit::Gaussian), &device);

        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let base_out = burn::tensor::module::linear(input.clone(), base_weight, base_bias);
        let lora_out = lora.forward(input);

        // Should match because scaling is 0 (LoRA contribution is zero)
        base_out
            .into_data()
            .assert_approx_eq::<FT>(&lora_out.into_data(), Tolerance::default());
    }

    /// Test the minimum valid rank of 1.
    #[test]
    fn test_lora_rank_one_minimum() {
        let device = Default::default();
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(1), &device);

        assert_eq!(lora.rank(), 1);
        assert_eq!(lora.lora_a.shape().dims::<2>(), [64, 1]);
        assert_eq!(lora.lora_b.shape().dims::<2>(), [1, 32]);

        // Should still work with rank 1
        let input = Tensor::<TestBackend, 2>::random([2, 64], Distribution::Default, &device);
        let output = lora.forward(input);
        assert_eq!(output.shape(), Shape::new([2, 32]));
    }

    /// Test adapter persistence with half precision settings.
    #[cfg(feature = "std")]
    #[test]
    fn test_adapter_half_precision_roundtrip() {
        use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};
        use std::fs;

        let device = Default::default();
        TestBackend::seed(&device, 42);

        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8).with_init(LoraInit::Gaussian), &device);

        let temp_dir = std::env::temp_dir();
        let adapter_path = temp_dir.join("test_lora_half_precision");
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();

        lora.save_adapter(&adapter_path, &recorder)
            .expect("Save failed");

        let fresh_lora = LinearConfig::new(64, 32)
            .init::<TestBackend>(&device)
            .with_lora(&LoraConfig::new(8), &device);
        let loaded = fresh_lora
            .load_adapter_file(&adapter_path, &recorder, &device)
            .expect("Load failed");

        // Verify dimensions preserved
        assert_eq!(loaded.rank(), 8);
        assert_eq!(loaded.d_input(), 64);
        assert_eq!(loaded.d_output(), 32);

        // Cleanup
        let _ = fs::remove_file(adapter_path.with_extension("mpk"));
    }

    /// Test forward pass with 4D input tensors.
    #[test]
    fn test_forward_4d_input() {
        let device = Default::default();
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(8), &device);

        // 4D input: [batch, seq, heads, d_input]
        let input = Tensor::<TestBackend, 4>::random([2, 4, 8, 64], Distribution::Default, &device);
        let output = lora.forward(input);
        assert_eq!(output.shape(), Shape::new([2, 4, 8, 32]));
    }

    /// Test forward pass with 1D input tensor (edge case).
    ///
    /// 1D input requires special handling since unsqueeze::<D>() on 2D LoRA weights
    /// fails when D=1. The forward method should upgrade to 2D, process, and squeeze back.
    #[test]
    fn test_lora_linear_1d_input() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let value = 2.;
        let config =
            LinearConfig::new(2, 3).with_initializer(crate::Initializer::Constant { value });
        let linear = config.init::<TestBackend>(&device);
        let lora = linear.with_lora(&LoraConfig::new(1), &device);

        // Create 1D and 2D inputs with the same data
        let input_1d = Tensor::<TestBackend, 1>::ones(Shape::new([2]), &device);
        let input_2d = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);

        // Both should produce equivalent results
        let result_1d = lora.forward(input_1d).unsqueeze::<2>();
        let result_2d = lora.forward(input_2d);

        result_1d
            .into_data()
            .assert_approx_eq::<FT>(&result_2d.into_data(), Tolerance::default());
    }

    /// Test LoRA formula with known values to verify mathematical correctness.
    ///
    /// LoRA computes: output = base(x) + (x @ A @ B) * scaling
    ///
    /// This test uses manually set weights to verify the exact formula:
    /// - base weight W = [[1, 2], [3, 4], [5, 6]] (3x2 -> d_input=3, d_output=2)
    /// - base bias = [0.1, 0.2]
    /// - LoRA A = [[0.1], [0.2], [0.3]] (3x1, rank=1)
    /// - LoRA B = [[0.5, 0.6]] (1x2)
    /// - alpha = 2.0, rank = 1, scaling = 2.0
    /// - input x = [[1, 1, 1]] (1x3)
    ///
    /// Expected:
    /// - base(x) = x @ W + bias = [1,1,1] @ [[1,2],[3,4],[5,6]] + [0.1,0.2]
    ///           = [1+3+5, 2+4+6] + [0.1, 0.2] = [9.1, 12.2]
    /// - lora(x) = (x @ A @ B) * scaling
    ///           = ([1,1,1] @ [[0.1],[0.2],[0.3]]) @ [[0.5, 0.6]] * 2.0
    ///           = [0.6] @ [[0.5, 0.6]] * 2.0 = [0.3, 0.36] * 2.0 = [0.6, 0.72]
    /// - output = base(x) + lora(x) = [9.1, 12.2] + [0.6, 0.72] = [9.7, 12.92]
    #[test]
    fn test_lora_formula_with_known_values() {
        use burn::tensor::TensorData;

        let device = Default::default();

        // Create linear layer with known weights
        // W: [d_input=3, d_output=2]
        let weight_data = TensorData::from([[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let weight = Tensor::<TestBackend, 2>::from_data(weight_data, &device);

        // bias: [d_output=2]
        let bias_data = TensorData::from([0.1_f32, 0.2]);
        let bias = Tensor::<TestBackend, 1>::from_data(bias_data, &device);

        // Build Linear manually
        let linear = Linear {
            weight: Param::from_tensor(weight),
            bias: Some(Param::from_tensor(bias)),
        };

        // Create LoRA config with known scaling
        let config = LoraConfig::new(1)
            .with_alpha(2.0)
            .with_init(LoraInit::Zeros);
        let mut lora = linear.with_lora(&config, &device);

        // Manually set LoRA A and B matrices
        // A: [d_input=3, rank=1]
        let lora_a_data = TensorData::from([[0.1_f32], [0.2], [0.3]]);
        let lora_a = Tensor::<TestBackend, 2>::from_data(lora_a_data, &device);
        lora.lora_a = Param::from_tensor(lora_a);

        // B: [rank=1, d_output=2]
        let lora_b_data = TensorData::from([[0.5_f32, 0.6]]);
        let lora_b = Tensor::<TestBackend, 2>::from_data(lora_b_data, &device);
        lora.lora_b = Param::from_tensor(lora_b);

        // Verify scaling is correct: alpha / rank = 2.0 / 1 = 2.0
        assert!((lora.scaling - 2.0).abs() < 1e-10);

        // Input: [batch=1, d_input=3]
        let input_data = TensorData::from([[1.0_f32, 1.0, 1.0]]);
        let input = Tensor::<TestBackend, 2>::from_data(input_data, &device);

        // Forward pass
        let output = lora.forward(input);

        // Expected output (calculated above):
        // base(x) = [9.1, 12.2]
        // lora(x) = [0.6, 0.72]
        // total = [9.7, 12.92]
        let expected_data = TensorData::from([[9.7_f32, 12.92]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected_data, Tolerance::default());
    }
}
