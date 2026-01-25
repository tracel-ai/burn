use crate::metric::{MetricAttributes, MetricName, SerializedEntry};

use super::super::{
    Metric, MetricMetadata,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::ElementConversion,
};
use core::marker::PhantomData;

/// Input type for the [PSNRMetric].
///
/// # Type Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions. Should be more than, or equal to 2 (default 4).
///   Dimension 0 is treated as batch, dimensions 1..D-1 are reduced to compute per-image MSE.
pub struct PSNRInput<B: Backend, const D: usize = 4> {
    /// Model outputs (predictions), as a tensor.
    outputs: Tensor<B, D>,
    /// Ground truth targets, as a tensor.
    targets: Tensor<B, D>,
}

impl<B: Backend, const D: usize> PSNRInput<B, D> {
    /// Creates a new PSNRInput with the given outputs and targets.
    ///
    /// # Arguments
    /// - `outputs`: The model outputs as a tensor.
    /// - `targets`: The ground truth targets as a tensor.
    ///
    /// # Returns
    /// A new instance of `PSNRInput`.
    ///
    /// # Panics
    /// - If `D` is less than 2.
    /// - If `outputs` and `targets` do not have the same shape.
    pub fn new(outputs: Tensor<B, D>, targets: Tensor<B, D>) -> Self {
        assert!(D >= 2, "PSNRInput requires at least 2 dimensions.");

        assert_eq!(
            outputs.dims(),
            targets.dims(),
            "Outputs and targets must have same shape"
        );

        Self { outputs, targets }
    }
}

/// Configuration for the [PSNRMetric].
#[derive(Debug, Clone, Copy)]
pub struct PSNRConfig {
    /// Maximum pixel value (1.0 for normalized images, 255.0 for 8-bit images).
    pub max_val: f64,
    /// Small value added to MSE to avoid numerical instability in log calculation.
    /// Should be small enough to allow high PSNR values but representable in f32 (default: 1e-12).
    pub epsilon: f64,
}

impl Default for PSNRConfig {
    fn default() -> Self {
        Self {
            max_val: 1.0,
            epsilon: 1e-12,
        }
    }
}

/// Peak Signal-to-Noise Ratio (PSNR) for evaluating image quality.
/// The PSNR is defined as:
/// `PSNR = 10 * log10(MAX² / MSE)`
/// where `MAX` is the maximum pixel value and `MSE` is the mean squared error.
///
/// # Type Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions. Should be more than, or equal to 2 (default 4).
///   Dimension 0 is treated as batch, dimensions 1..D-1 are reduced to compute per-image MSE.
#[derive(Default, Clone)]
pub struct PSNRMetric<B: Backend, const D: usize = 4> {
    name: MetricName,
    state: NumericMetricState,
    _b: PhantomData<B>,
    config: PSNRConfig,
}

impl<B: Backend, const D: usize> PSNRMetric<B, D> {
    /// Creates a new PSNR metric instance with default config.
    pub fn new() -> Self {
        Self::with_config(PSNRConfig::default())
    }

    /// Creates a new PSNR metric with a custom config.
    pub fn with_config(config: PSNRConfig) -> Self {
        let name = MetricName::new(format!("PSNR ({}D)", D));
        Self {
            name,
            state: NumericMetricState::default(),
            _b: PhantomData,
            config,
        }
    }
}

impl<B: Backend, const D: usize> Metric for PSNRMetric<B, D> {
    type Input = PSNRInput<B, D>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        if item.outputs.dims() != item.targets.dims() {
            panic!(
                "Outputs and targets must have the same dimensions. Got {:?} and {:?}",
                item.outputs.dims(),
                item.targets.dims()
            );
        }

        // Dimension 0 is treated as batch size
        let batch_size = item.outputs.dims()[0];

        // Compute element-wise squared differences: (outputs - targets)^2
        let diff = item.outputs.clone() - item.targets.clone();
        let squared_diff = diff.powf_scalar(2.0);

        // Compute per-image MSE by reducing over all dimensions except batch (dim 0)
        // For 4D tensors [B, C, H, W], this reduces over dimensions [1, 2, 3] (C, H, W)
        let axes: Vec<usize> = (1..D).collect();
        let mse_per_image = squared_diff.mean_dims(&axes);

        // Reshape from [B, 1, 1, 1] to [B] for easier per-image processing
        let mse_flat = mse_per_image.reshape([batch_size]);

        // Compute PSNR: 10 * log10(MAX^2 / MSE)
        let max_val_sq = self.config.max_val.powi(2);
        let ln_10 = 10.0_f64.ln();

        // Add epsilon to MSE to avoid division by zero and log(0) when MSE is exactly 0
        let mse_safe = mse_flat.clone() + self.config.epsilon;

        // Convert natural log to log10: log10(x) = ln(x) / ln(10)
        let psnr_per_image = mse_safe
            .recip() // 1 / (MSE + epsilon)
            .mul_scalar(max_val_sq) // MAX^2 / (MSE + epsilon)
            .log() // ln(MAX^2 / (MSE + epsilon))
            .div_scalar(ln_10) // log10(MAX^2 / (MSE + epsilon))
            .mul_scalar(10.0); // 10 * log10(MAX^2 / (MSE + epsilon))

        // Average PSNR across batch and update metric state
        let avg_psnr = psnr_per_image.mean().into_scalar().elem::<f64>();

        self.state.update(
            avg_psnr,
            batch_size,
            FormatOptions::new(self.name()).precision(2).unit("dB"),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn attributes(&self) -> MetricAttributes {
        crate::metric::NumericAttributes {
            unit: Some("dB".into()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend, const D: usize> crate::metric::Numeric for PSNRMetric<B, D> {
    fn value(&self) -> crate::metric::NumericEntry {
        self.state.current_value()
    }
    fn running_value(&self) -> crate::metric::NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestBackend, metric::Numeric};
    use burn_core::tensor::Tensor;

    #[test]
    fn test_psnr_perfect_match() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();
        let input = PSNRInput::new(
            Tensor::from_data([[[[1.0, 0.5], [0.8, 0.2]]]], &device),
            Tensor::from_data([[[[1.0, 0.5], [0.8, 0.2]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();
        assert!(
            psnr > 100.0,
            "Perfect match should have very high PSNR, got {}",
            psnr
        );
    }

    #[test]
    fn test_psnr_small_error() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();
        let input = PSNRInput::new(
            Tensor::from_data([[[[1.0, 0.5], [0.8, 0.2]]]], &device),
            Tensor::from_data([[[[0.99, 0.51], [0.81, 0.19]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();
        assert!(
            psnr > 20.0 && psnr < 100.0,
            "Small error should give moderate PSNR, got {}",
            psnr
        );
    }

    #[test]
    fn test_psnr_large_error() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();
        let input = PSNRInput::new(
            Tensor::from_data([[[[1.0, 0.0], [0.0, 1.0]]]], &device),
            Tensor::from_data([[[[0.0, 1.0], [1.0, 0.0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();
        assert!(
            psnr < 20.0,
            "Large error should give low PSNR, got {}",
            psnr
        );
    }

    #[test]
    fn test_psnr_batch() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();
        let input = PSNRInput::new(
            Tensor::from_data(
                [[[[1.0, 0.5], [0.8, 0.2]]], [[[0.5, 1.0], [0.2, 0.8]]]],
                &device,
            ),
            Tensor::from_data(
                [[[[1.0, 0.5], [0.8, 0.2]]], [[[0.5, 1.0], [0.2, 0.8]]]],
                &device,
            ),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();
        assert!(
            psnr > 100.0,
            "Perfect batch match should have very high PSNR, got {}",
            psnr
        );
    }

    #[test]
    fn test_psnr_custom_max_val() {
        let device = Default::default();
        let config = PSNRConfig {
            max_val: 255.0,
            epsilon: 1e-8,
        };
        let mut metric = PSNRMetric::<TestBackend, 4>::with_config(config);
        let input = PSNRInput::new(
            Tensor::from_data([[[[255.0, 128.0], [200.0, 50.0]]]], &device),
            Tensor::from_data([[[[255.0, 128.0], [200.0, 50.0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();
        assert!(
            psnr > 100.0,
            "Perfect match with max_val=255 should have very high PSNR, got {}",
            psnr
        );
    }

    #[test]
    #[should_panic(expected = "Outputs and targets must have same shape")]
    fn test_psnr_shape_mismatch_panics() {
        let device = Default::default();
        let _input = PSNRInput::<TestBackend, 4>::new(
            Tensor::<TestBackend, 4>::from_data([[[[1.0, 0.5]]]], &device),
            Tensor::<TestBackend, 4>::from_data([[[[1.0, 0.5], [0.8, 0.2]]]], &device),
        );
    }

    #[test]
    fn test_psnr_3d_input() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 3>::new();

        let input = PSNRInput::new(
            Tensor::from_data([[[1.0, 0.5], [0.8, 0.2]]], &device),
            Tensor::from_data([[[1.0, 0.5], [0.8, 0.2]]], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();

        assert!(
            psnr > 100.0,
            "Perfect match in 3D should have very high PSNR"
        );
    }

    #[test]
    fn test_psnr_zero_mse_is_finite() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();

        let input = PSNRInput::new(
            Tensor::zeros([1, 1, 2, 2], &device),
            Tensor::zeros([1, 1, 2, 2], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let psnr = metric.value().current();

        assert!(psnr.is_finite(), "PSNR should be finite for zero MSE");
        assert!(psnr > 100.0);
    }

    #[test]
    fn test_psnr_monotonicity() {
        let device = Default::default();

        let mut metric_small = PSNRMetric::<TestBackend, 4>::new();
        let mut metric_large = PSNRMetric::<TestBackend, 4>::new();

        let perfect = Tensor::from_data([[[[1.0, 1.0], [1.0, 1.0]]]], &device);

        let small_error = Tensor::from_data([[[[0.99, 1.01], [1.0, 1.0]]]], &device);
        let large_error = Tensor::from_data([[[[0.5, 1.5], [0.0, 2.0]]]], &device);

        metric_small.update(
            &PSNRInput::new(perfect.clone(), small_error),
            &MetricMetadata::fake(),
        );

        metric_large.update(
            &PSNRInput::new(perfect, large_error),
            &MetricMetadata::fake(),
        );

        let psnr_small = metric_small.value().current();
        let psnr_large = metric_large.value().current();

        assert!(
            psnr_small > psnr_large,
            "Smaller error should yield higher PSNR"
        );
    }

    #[test]
    fn test_psnr_mixed_batch() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();

        let outputs = Tensor::from_data(
            [[[[1.0, 1.0], [1.0, 1.0]]], [[[0.0, 0.0], [0.0, 0.0]]]],
            &device,
        );

        let targets = Tensor::from_data(
            [[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 1.0], [1.0, 1.0]]]],
            &device,
        );

        metric.update(&PSNRInput::new(outputs, targets), &MetricMetadata::fake());

        let psnr = metric.value().current();

        assert!(
            psnr > 10.0 && psnr < 100.0,
            "Mixed batch should average PSNR"
        );
    }

    #[test]
    fn test_psnr_running_average() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();

        let good = PSNRInput::new(
            Tensor::from_data([[[[1.0, 1.0]]]], &device),
            Tensor::from_data([[[[1.0, 1.0]]]], &device),
        );

        let bad = PSNRInput::new(
            Tensor::from_data([[[[0.0, 0.0]]]], &device),
            Tensor::from_data([[[[1.0, 1.0]]]], &device),
        );

        metric.update(&good, &MetricMetadata::fake());
        let first = metric.value().current();

        metric.update(&bad, &MetricMetadata::fake());
        let running = metric.running_value().current();

        assert!(
            running < first,
            "Running average should decrease after worse batch"
        );
    }

    #[test]
    fn test_psnr_clear_resets_state() {
        let device = Default::default();
        let mut metric = PSNRMetric::<TestBackend, 4>::new();

        let input = PSNRInput::new(
            Tensor::from_data([[[[1.0, 1.0]]]], &device),
            Tensor::from_data([[[[1.0, 1.0]]]], &device),
        );

        metric.update(&input, &MetricMetadata::fake());
        assert!(metric.value().current() > 0.0);

        metric.clear();
        let cleared = metric.value().current();

        assert!(
            cleared.is_nan(),
            "Clearing should reset metric state to NaN"
        );
    }
}
