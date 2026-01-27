use crate::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::ElementConversion,
};
use core::marker::PhantomData;
use std::f64::consts::LN_10;

/// Input type for the [PsnrMetric].
///
/// Both tensors must have shape `[N, C, H, W]`:
/// - `N`: Batch size
/// - `C`: Number of channels (1 for grayscale, 3 for RGB, etc.)
/// - `H`: Height
/// - `W`: Width
pub struct PsnrInput<B: Backend> {
    /// Model output (predictions/reconstructions) images with shape `[N, C, H, W]`.
    outputs: Tensor<B, 4>,
    /// Ground truth images with shape `[N, C, H, W]`.
    targets: Tensor<B, 4>,
}

impl<B: Backend> PsnrInput<B> {
    /// Creates a new PsnrInput with the given outputs and targets.
    ///
    /// Inputs are expected to have the dimensions `[N, C, H, W]`
    /// where `N` is the batch size, `C` is the number of channels,
    /// `H` is the height of the image, and `W` is the width of the image.
    ///
    /// # Arguments
    /// - `outputs`: The model output images with shape `[N, C, H, W]`.
    /// - `targets`: The ground truth images with shape `[N, C, H, W]`.
    ///
    /// # Returns
    /// A new instance of `PsnrInput`.
    ///
    /// # Panics
    /// - If `outputs` and `targets` do not have the same shape.
    pub fn new(outputs: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        assert!(
            outputs.dims() == targets.dims(),
            "Shape mismatch: outputs {:?}, targets {:?}",
            outputs.dims(),
            targets.dims()
        );
        Self { outputs, targets }
    }
}

/// Configuration for the [PsnrMetric].
#[derive(Debug, Clone, Copy)]
pub struct PsnrMetricConfig {
    /// Maximum possible pixel value.
    /// - Use `1.0` for normalized images in range \[0, 1\]
    /// - Use `255.0` for 8-bit images in range \[0, 255\]
    pub max_pixel_val: f64,
    /// Epsilon value for numerical stability when MSE is very small or zero.
    ///
    /// When MSE falls below this threshold, it is clamped to `epsilon`,
    /// resulting in a maximum PSNR of approximately `10 * log10(max_pixel_val² / epsilon)` dB.
    ///
    /// Default is `1e-10`, which yields ~100 dB for perfect reconstruction with `max_pixel_val = 1.0`.
    pub epsilon: f64,
}

impl PsnrMetricConfig {
    /// Creates a configuration with the specified maximum pixel value.
    ///
    /// # Example
    /// ```ignore
    /// // Normalized images [0, 1]
    /// let config = PsnrMetricConfig::new(1.0);
    ///
    /// // 8-bit images [0, 255]  
    /// let config = PsnrMetricConfig::new(255.0);
    /// // Also set a custom epsilon value
    /// let config = PsnrMetricConfig::new(255.0).with_epsilon(1e-8);
    /// ```
    pub fn new(max_pixel_val: f64) -> Self {
        assert!(max_pixel_val > 0.0, "max_pixel_val must be positive");
        Self {
            max_pixel_val,
            epsilon: 1e-10,
        }
    }

    /// Sets a custom epsilon for numerical stability near zero MSE
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        self.epsilon = epsilon;
        self
    }
}

/// The peak signal-to-noise ratio (PSNR) metric for image quality assessment.
///
/// PSNR is commonly used to measure the quality of reconstructed images
/// compared to the original. Higher values (in dB) indicate better quality.
///
/// # Formula
/// ```text
/// PSNR = 10 * log10(MAX^2 / MSE)
/// ```
/// where MAX is the maximum possible pixel value and MSE is the mean squared error.
///
/// # Note
/// - PSNR is computed for each image first, and then it is averaged across all the images in the batch.
/// - For perfect reconstruction (MSE = 0), the MSE is clamped to `epsilon` to avoid division by zero,
///   yielding a maximum PSNR of `10 * log10(MAX^2 / epsilon)` dB.
#[derive(Clone)]
pub struct PsnrMetric<B: Backend> {
    name: MetricName,
    /// Internal state for numeric metric aggregation.
    state: NumericMetricState,
    /// Marker for backend type.
    _b: PhantomData<B>,
    /// Configuration for the metric.
    config: PsnrMetricConfig,
}

impl<B: Backend> PsnrMetric<B> {
    /// Creates a new PSNR metric with the given configuration.
    ///
    /// # Example
    /// ```ignore
    /// let config = PsnrMetricConfig::new(1.0);
    /// let metric = PsnrMetric::<B>::new(config);
    /// ```
    pub fn new(config: PsnrMetricConfig) -> Self {
        Self {
            name: MetricName::new(format!("PSNR@{}", config.max_pixel_val)),
            state: NumericMetricState::default(),
            config,
            _b: PhantomData,
        }
    }

    /// Overrides the default metric name which is `PSNR@{max_pixel_val}`.
    ///
    /// Examples names:
    /// - `PSNR@1.0`
    /// - `PSNR@255.0`
    ///
    /// Use this method to provide a custom name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = MetricName::new(name.to_string());
        self
    }
}

impl<B: Backend> Metric for PsnrMetric<B> {
    type Input = PsnrInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let dims = item.outputs.dims();
        let batch_size = dims[0];
        let outputs = item.outputs.clone();
        let targets = item.targets.clone();

        // Compute per-image MSE by reducing over all dimensions except batch (dims 1, 2, 3)
        // Resulting shape: [N, 1, 1, 1]
        let diff = outputs.sub(targets);
        let mse_per_image = diff.powi_scalar(2).mean_dims(&[1, 2, 3]);
        // Flatten to shape: [N]
        let mse_flat = mse_per_image.flatten::<1>(0, 3);
        // Clamp MSE to avoid division by 0 in the expression (MAX^2 / MSE)
        let mse_clamped = mse_flat.clamp_min(self.config.epsilon);
        let max_squared = self.config.max_pixel_val * self.config.max_pixel_val;

        // Compute PSNR for each image and accumulate
        // PSNR value in dB (using the change of base formula):
        // 10 * log10(MAX^2 / MSE) = 10 * ln(MAX^2 / MSE) / ln(10)
        //                         = ln(MAX^2 / MSE) * (10 / ln(10))
        let psnr_per_image = mse_clamped
            .recip()
            .mul_scalar(max_squared)
            .log()
            .mul_scalar(10.0 / LN_10);
        let avg_psnr = psnr_per_image.mean().into_scalar().elem::<f64>();

        self.state.update(
            avg_psnr,
            batch_size,
            FormatOptions::new(self.name()).unit("dB").precision(2),
        )
    }

    /// Clears the metric state.
    fn clear(&mut self) {
        self.state.reset();
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("dB".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for PsnrMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestBackend, metric::Numeric};
    use burn_core::tensor::TensorData;

    #[test]
    fn test_psnr_perfect_reconstruction() {
        // When outputs exactly match targets, PSNR should be very high
        // (limited by epsilon clamping to ~100 dB with default epsilon=1e-10)
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0_f32, 0.5], [0.25, 0.75]]]]),
            &device,
        );
        let targets = outputs.clone();

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        // With epsilon = 1e-10 and max=1.0:
        // PSNR = 10 * log10(1.0 / 1e-10) = 100 dB
        let psnr = metric.value().current();
        assert!(
            psnr >= 99.0,
            "PSNR for perfect reconstruction should be ~100 dB, got {} dB",
            psnr
        );
    }

    #[test]
    fn test_psnr_constant_error() {
        // Constant error of 0.1 across all pixels
        // MSE = 0.01, PSNR = 10 * log10(1.0 / 0.01) = 20 dB
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1_f32, 0.1], [0.1, 0.1]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0_f32, 0.0], [0.0, 0.0]]]]),
            &device,
        );

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        assert!(
            (psnr - 20.0).abs() < 0.01,
            "Expected PSNR ~20 dB, got {} dB",
            psnr
        );
    }

    #[test]
    fn test_psnr_varying_error() {
        // Errors: 0.1, 0.2, 0.3, 0.4 → squared: 0.01, 0.04, 0.09, 0.16
        // MSE = 0.075, PSNR = 10 * log10(1.0 / 0.075) ≈ 11.249 dB
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1_f32, 0.2], [0.3, 0.4]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0_f32, 0.0], [0.0, 0.0]]]]),
            &device,
        );

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 10.0 * (1.0_f64 / 0.075).log10();
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{:.3} dB, got {} dB",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_max_pixel_255() {
        // Test with 8-bit image range [0, 255]
        // Error = 10 everywhere, MSE = 100
        // PSNR = 10 * log10(255^2 / 100) ≈ 28.13 dB
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[10.0_f32, 10.0], [10.0, 10.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0_f32, 0.0], [0.0, 0.0]]]]),
            &device,
        );

        let config = PsnrMetricConfig::new(255.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 10.0 * (255.0_f64 * 255.0 / 100.0).log10();
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{:.3} dB, got {} dB",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_batch_averaging() {
        // Batch of 2 images with different MSEs
        // Image 1: error 0.1 → MSE = 0.01 → PSNR = 20 dB
        // Image 2: error 0.01 → MSE = 0.0001 → PSNR = 40 dB
        // Average PSNR = 30 dB
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.1_f32, 0.1], [0.1, 0.1]]],
                [[[0.01_f32, 0.01], [0.01, 0.01]]],
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.0_f32, 0.0], [0.0, 0.0]]],
                [[[0.0_f32, 0.0], [0.0, 0.0]]],
            ]),
            &device,
        );

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 30.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected average PSNR ~{} dB, got {} dB",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_multichannel() {
        // Test with 3 channels (RGB-like)
        // All channels have constant error 0.1 → MSE = 0.01 → PSNR = 20 dB
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [[0.1_f32, 0.1], [0.1, 0.1]],
                [[0.1_f32, 0.1], [0.1, 0.1]],
                [[0.1_f32, 0.1], [0.1, 0.1]],
            ]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::zeros([1, 3, 2, 2], &device);

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 20.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{} dB, got {} dB",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_running_average() {
        // Test running average across multiple updates
        let device = Default::default();
        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);

        // First update: error 0.1 → MSE = 0.01 → PSNR = 20 dB
        let outputs1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1_f32, 0.1], [0.1, 0.1]]]]),
            &device,
        );
        let targets1 = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);
        let input1 = PsnrInput::new(outputs1, targets1);
        let _entry = metric.update(&input1, &MetricMetadata::fake());

        let psnr1 = metric.value().current();
        let expected_psnr1 = 20.0;
        assert!(
            (psnr1 - expected_psnr1).abs() < 0.01,
            "First update PSNR should be ~{} dB, got {} dB",
            expected_psnr1,
            psnr1
        );

        // Second update: error 0.01 → MSE = 0.0001 → PSNR = 40 dB
        let outputs2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.01_f32, 0.01], [0.01, 0.01]]]]),
            &device,
        );
        let targets2 = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);
        let input2 = PsnrInput::new(outputs2, targets2);
        let _entry = metric.update(&input2, &MetricMetadata::fake());

        // Running average: (20 + 40) / 2 = 30 dB
        let running_avg_psnr = metric.running_value().current();
        let expected_running_avg_psnr = 30.0;
        assert!(
            (running_avg_psnr - expected_running_avg_psnr).abs() < 0.01,
            "Running average should be ~{} dB, got {} dB",
            expected_running_avg_psnr,
            running_avg_psnr
        );
    }

    #[test]
    fn test_psnr_clear() {
        // Error 0.1 → MSE = 0.01 → PSNR = 20 dB
        let device = Default::default();
        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);

        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1_f32, 0.1], [0.1, 0.1]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 20.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{} dB, got {} dB",
            expected_psnr,
            psnr
        );

        // Clear and verify reset
        metric.clear();
        let psnr = metric.running_value().current();
        assert!(psnr.is_nan(), "Expected NaN after clear, got {} dB", psnr)
    }

    #[test]
    fn test_psnr_custom_name() {
        let config = PsnrMetricConfig::new(1.0);
        let metric = PsnrMetric::<TestBackend>::new(config).with_name("CustomPSNR");

        assert_eq!(metric.name().to_string(), "CustomPSNR");
    }

    #[test]
    fn test_psnr_custom_epsilon() {
        let device = Default::default();
        // With a larger epsilon, perfect reconstruction gives lower PSNR
        let config = PsnrMetricConfig::new(1.0).with_epsilon(0.01);
        let mut metric = PsnrMetric::<TestBackend>::new(config);

        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.5_f32, 0.5], [0.5, 0.5]]]]),
            &device,
        );
        let targets = outputs.clone();
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        // With epsilon = 0.01, PSNR = 10 * log10(1.0 / 0.01) = 20 dB
        let psnr = metric.value().current();
        let expected_psnr = 20.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{} dB with epsilon=0.01, got {}",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_negative_errors() {
        // Test that negative differences (target > output) work correctly
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0_f32, 0.0], [0.0, 0.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1_f32, 0.1], [0.1, 0.1]]]]),
            &device,
        );

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        // Same MSE as positive errors (0.01), so PSNR = 20 dB
        let psnr = metric.value().current();
        let expected_psnr = 20.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{} dB, got {}",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_large_batch() {
        // Test with a larger batch to verify batch dimension handling
        let device = Default::default();
        let batch_size = 8;

        // All images have constant error 0.1 → MSE = 0.01 → PSNR = 20 dB
        let outputs = Tensor::<TestBackend, 4>::full([batch_size, 3, 4, 4], 0.1, &device);
        let targets = Tensor::<TestBackend, 4>::zeros([batch_size, 3, 4, 4], &device);

        let config = PsnrMetricConfig::new(1.0);
        let mut metric = PsnrMetric::<TestBackend>::new(config);
        let input = PsnrInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let psnr = metric.value().current();
        let expected_psnr = 20.0;
        assert!(
            (psnr - expected_psnr).abs() < 0.01,
            "Expected PSNR ~{} dB, got {}",
            expected_psnr,
            psnr
        );
    }

    #[test]
    fn test_psnr_attributes() {
        let config = PsnrMetricConfig::new(1.0);
        let metric = PsnrMetric::<TestBackend>::new(config);
        let attrs = metric.attributes();

        match attrs {
            MetricAttributes::Numeric(numeric_attrs) => {
                assert_eq!(numeric_attrs.unit, Some("dB".to_string()));
                assert!(numeric_attrs.higher_is_better);
            }
            _ => panic!("Expected numeric attributes"),
        }
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_psnr_shape_mismatch() {
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);
        let targets = Tensor::<TestBackend, 4>::zeros([1, 1, 3, 3], &device);

        let _ = PsnrInput::new(outputs, targets);
    }

    #[test]
    #[should_panic(expected = "max_pixel_val must be positive")]
    fn test_psnr_negative_max_pixel_val() {
        let _ = PsnrMetricConfig::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "max_pixel_val must be positive")]
    fn test_psnr_zero_max_pixel_val() {
        let _ = PsnrMetricConfig::new(0.0);
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn test_psnr_negative_epsilon() {
        let _ = PsnrMetricConfig::new(1.0).with_epsilon(-1e-10);
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn test_psnr_zero_epsilon() {
        let _ = PsnrMetricConfig::new(1.0).with_epsilon(0.0);
    }
}
