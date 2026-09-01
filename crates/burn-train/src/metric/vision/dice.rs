use crate::metric::{MetricAttributes, MetricName, NumericEntry, SerializedEntry, format_float};

use super::super::{Metric, MetricMetadata, state::FormatOptions};
use burn_core::{
    prelude::Tensor,
    tensor::{Int, s},
};

/// Input type for the [DiceMetric].
///
/// # Type Parameters
/// - `D`: Number of dimensions. Should be more than, or equal to 3 (default 4).
pub struct DiceInput<const D: usize = 4> {
    /// Model outputs (predictions), as a tensor.
    outputs: Tensor<D, Int>,
    /// Ground truth targets, as a tensor.
    targets: Tensor<D, Int>,
}

impl<const D: usize> DiceInput<D> {
    /// Creates a new DiceInput with the given outputs and targets.
    ///
    /// Inputs are expected to have the dimensions `[B, C, ...]`
    /// where `B` is the batch size, `C` is the number of classes,
    /// and `...` represents additional dimensions (e.g., height, width for images).
    ///
    /// If `C` is more than 1, the first class (index 0) is considered the background.
    /// Additionally, one-hot encoding is the responsibility of the caller.
    ///
    /// # Arguments
    /// - `outputs`: The model outputs as a tensor.
    /// - `targets`: The ground truth targets as a tensor.
    ///
    /// # Returns
    /// A new instance of `DiceInput`.
    ///
    ///  # Panics
    /// - If `D` is less than 3.
    /// - If `outputs` and `targets` do not have the same dimensions.
    /// - If `outputs` or `targets` do not have exactly `D` dimensions.
    /// - If `outputs` and `targets` do not have the same shape.
    pub fn new(outputs: Tensor<D, Int>, targets: Tensor<D, Int>) -> Self {
        assert!(D >= 3, "DiceInput requires at least 3 dimensions.");
        assert!(
            outputs.dims() == targets.dims(),
            "Outputs and targets must have the same dimensions. Got {:?} and {:?}",
            outputs.dims(),
            targets.dims()
        );
        Self { outputs, targets }
    }
}

/// Configuration for the [DiceMetric].
#[derive(Debug, Clone, Copy)]
pub struct DiceMetricConfig {
    /// Epsilon added to the numerator and denominator to avoid division by zero.
    /// This defines two empty masks as a perfect match with a Dice score of `1.0`.
    pub epsilon: f64,
    /// Whether to include the background class in the metric calculation.
    /// The background is assumed to be the first class (index 0).
    /// if `true`, will panic if there are fewer than 2 classes.
    pub include_background: bool,
}

impl Default for DiceMetricConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            include_background: false,
        }
    }
}

/// Accumulates the raw Dice numerator and denominator across batches.
///
/// Averaging per-batch Dice coefficients is biased when batches contain different amounts of
/// foreground. Keeping the raw statistics allows the running and final values to be computed over
/// the complete epoch instead.
#[derive(Clone)]
struct DiceMetricState {
    numerator: f64,
    denominator: f64,
    count: usize,
    current: f64,
    current_count: usize,
}

impl DiceMetricState {
    fn new() -> Self {
        Self {
            numerator: 0.0,
            denominator: 0.0,
            count: 0,
            current: f64::NAN,
            current_count: 0,
        }
    }

    fn dice(numerator: f64, denominator: f64, epsilon: f64) -> f64 {
        (numerator + epsilon) / (denominator + epsilon)
    }

    fn update(
        &mut self,
        intersection: f64,
        outputs_sum: f64,
        targets_sum: f64,
        batch_size: usize,
        epsilon: f64,
    ) {
        let numerator = 2.0 * intersection;
        let denominator = outputs_sum + targets_sum;

        self.numerator += numerator;
        self.denominator += denominator;
        self.count += batch_size;
        self.current = Self::dice(numerator, denominator, epsilon);
        self.current_count = batch_size;
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn running(&self, epsilon: f64) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            Self::dice(self.numerator, self.denominator, epsilon)
        }
    }

    fn compute_update(&self, format: FormatOptions, epsilon: f64) -> SerializedEntry {
        self.compute(format, epsilon, false)
    }

    fn compute_final(&self, format: FormatOptions, epsilon: f64) -> SerializedEntry {
        self.compute(format, epsilon, true)
    }

    fn compute(&self, format: FormatOptions, epsilon: f64, final_entry: bool) -> SerializedEntry {
        let running = self.running(epsilon);
        let serialized = if final_entry {
            NumericEntry::Final(running).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.current,
                count: self.current_count,
            }
            .serialize()
        };

        let (formatted_current, formatted_running) = match format.precision_value() {
            Some(precision) => (
                format_float(self.current, precision),
                format_float(running, precision),
            ),
            None => (format!("{}", self.current), format!("{running}")),
        };

        SerializedEntry::new(
            format!("epoch {formatted_running} - batch {formatted_current}"),
            serialized,
        )
    }

    fn current_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.current,
            count: self.current_count,
        }
    }

    fn running_value(&self, epsilon: f64) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value: self.running(epsilon),
            count: self.count,
        }
    }

    fn final_value(&self, epsilon: f64) -> NumericEntry {
        NumericEntry::Final(self.running(epsilon))
    }
}

impl Default for DiceMetricState {
    fn default() -> Self {
        Self::new()
    }
}

/// The Dice-Sorenson coefficient (DSC) for evaluating overlap between two binary masks.
/// The DSC is defined as:
/// `DSC = 2 * (|X ∩ Y|) / (|X| + |Y|)`
/// where `X` is the model output and `Y` is the ground truth target.
///
/// # Aggregation
///
/// The batch value is computed globally over all samples and classes in the batch. Running and
/// final values use global micro-aggregation over the epoch: batch numerators and denominators are
/// summed before their ratio is calculated.
///
///  # Type Parameters
/// - `D`: Number of dimensions. Should be more than, or equal to 3 (default 4).
#[derive(Default, Clone)]
pub struct DiceMetric<const D: usize = 4> {
    name: MetricName,
    /// Internal state for raw Dice statistic aggregation.
    state: DiceMetricState,
    /// Configuration for the metric.
    config: DiceMetricConfig,
}

impl<const D: usize> DiceMetric<D> {
    /// Creates a new Dice metric instance with default config.
    pub fn new() -> Self {
        Self::with_config(DiceMetricConfig::default())
    }

    /// Creates a new Dice metric with a custom config.
    pub fn with_config(config: DiceMetricConfig) -> Self {
        let name = MetricName::new(format!("{D}D Dice Metric"));
        assert!(D >= 3, "DiceMetric requires at least 3 dimensions.");
        assert!(
            config.epsilon.is_finite() && config.epsilon > 0.0,
            "epsilon must be finite and positive"
        );
        Self {
            name,
            config,
            ..Default::default()
        }
    }
}

impl<const D: usize> Metric for DiceMetric<D> {
    type Input = DiceInput<D>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        // Dice coefficient: 2 * (|X ∩ Y|) / (|X| + |Y|)
        if item.outputs.dims() != item.targets.dims() {
            panic!(
                "Outputs and targets must have the same dimensions. Got {:?} and {:?}",
                item.outputs.dims(),
                item.targets.dims()
            );
        }

        let dims = item.outputs.dims();
        let batch_size = dims[0];
        let n_classes = dims[1];

        let mut outputs = item.outputs.clone();
        let mut targets = item.targets.clone();

        if !self.config.include_background && n_classes > 1 {
            // If not including background, we can ignore the first class
            outputs = outputs.slice(s![.., 1..]);
            targets = targets.slice(s![.., 1..]);
        } else if self.config.include_background && n_classes < 2 {
            // If including background, we need at least 2 classes
            panic!("Dice metric requires at least 2 classes when including background.");
        }

        let intersection = (outputs.clone() * targets.clone()).sum();
        let outputs_sum = outputs.sum();
        let targets_sum = targets.sum();

        // Convert to f64
        let intersection_val = intersection.into_scalar::<f64>();
        let outputs_sum_val = outputs_sum.into_scalar::<f64>();
        let targets_sum_val = targets_sum.into_scalar::<f64>();

        self.state.update(
            intersection_val,
            outputs_sum_val,
            targets_sum_val,
            batch_size,
            self.config.epsilon,
        );
        self.state.compute_update(
            FormatOptions::new(self.name()).precision(4),
            self.config.epsilon,
        )
    }

    fn compute(&mut self) -> SerializedEntry {
        self.state.compute_final(
            FormatOptions::new(self.name()).precision(4),
            self.config.epsilon,
        )
    }

    /// Clears the metric state.
    fn clear(&mut self) {
        self.state.reset();
    }

    fn attributes(&self) -> MetricAttributes {
        crate::metric::NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<const D: usize> crate::metric::Numeric for DiceMetric<D> {
    fn value(&self) -> Option<crate::metric::NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<crate::metric::NumericEntry> {
        Some(self.state.running_value(self.config.epsilon))
    }

    fn final_value(&self) -> crate::metric::NumericEntry {
        self.state.final_value(self.config.epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Numeric;
    use burn_core::tensor::{Shape, Tensor};
    use rstest::rstest;

    #[test]
    fn test_dice_perfect_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_no_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
            Tensor::from_data([[[[0, 1], [0, 1]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!(metric.value().unwrap().current() < 1e-6);
    }

    #[test]
    fn test_dice_partial_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 1], [0, 0]]]], &device),
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        // intersection = 1, sum = 2+2=4, dice = 2*1/4 = 0.5
        assert!((metric.value().unwrap().current() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dice_epoch_accumulates_raw_statistics() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();

        // A perfect batch with one foreground pixel.
        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[1]]]], &device),
                Tensor::from_data([[[[1]]]], &device),
            ),
            &MetricMetadata::fake(),
        );

        // A batch with four false-positive foreground pixels.
        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[1, 1, 1, 1]]]], &device),
                Tensor::from_data([[[[0, 0, 0, 0]]]], &device),
            ),
            &MetricMetadata::fake(),
        );

        // Global Dice: (2 * 1 + epsilon) / (5 + 1 + epsilon) = 1 / 3.
        // Averaging the two batch scores would incorrectly produce approximately 1 / 2.
        let expected = (2.0 + metric.config.epsilon) / (6.0 + metric.config.epsilon);
        let current = metric.value().unwrap().current();
        let running = metric.running_value().unwrap().current();
        let computed = NumericEntry::deserialize(&metric.compute().serialized)
            .unwrap()
            .current();
        let final_value = metric.final_value().current();

        assert!(current < 1e-6);
        assert!((running - expected).abs() < 1e-6);
        assert!((computed - expected).abs() < 1e-6);
        assert!((final_value - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dice_epoch_matches_pytorch_with_varying_batch_sizes() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();

        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[1, 0, 1, 0]]]], &device),
                Tensor::from_data([[[[1, 0, 0, 0]]]], &device),
            ),
            &MetricMetadata::fake(),
        );
        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[1]]], [[[1]]], [[[1]]]], &device),
                Tensor::from_data([[[[0]]], [[[1]]], [[[0]]]], &device),
            ),
            &MetricMetadata::fake(),
        );

        // PyTorch global-micro reference: (2 * 2 + epsilon) / (5 + 2 + epsilon).
        let expected = (4.0 + metric.config.epsilon) / (7.0 + metric.config.epsilon);
        let running = metric.running_value().unwrap().current();

        assert!((running - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dice_epoch_empty_masks() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();

        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[0, 0]]]], &device),
                Tensor::from_data([[[[0, 0]]]], &device),
            ),
            &MetricMetadata::fake(),
        );
        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[0, 0, 0]]], [[[0, 0, 0]]]], &device),
                Tensor::from_data([[[[0, 0, 0]]], [[[0, 0, 0]]]], &device),
            ),
            &MetricMetadata::fake(),
        );

        // Burn's epsilon convention defines two empty masks as a perfect match.
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
        assert!((metric.running_value().unwrap().current() - 1.0).abs() < 1e-6);
        assert!((metric.final_value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_epoch_excludes_background() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();

        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[1, 1, 0, 0]], [[0, 1, 1, 0]]]], &device),
                Tensor::from_data([[[[1, 0, 1, 0]], [[0, 1, 0, 1]]]], &device),
            ),
            &MetricMetadata::fake(),
        );
        metric.update(
            &DiceInput::new(
                Tensor::from_data([[[[0, 0]], [[1, 1]]]], &device),
                Tensor::from_data([[[[1, 1]], [[1, 1]]]], &device),
            ),
            &MetricMetadata::fake(),
        );

        // Only class 1 is included: (2 * 3 + epsilon) / (4 + 4 + epsilon).
        let expected = (6.0 + metric.config.epsilon) / (8.0 + metric.config.epsilon);
        let running = metric.running_value().unwrap().current();

        assert!((running - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dice_clear_resets_epoch_statistics() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1]]]], &device),
            Tensor::from_data([[[[1]]]], &device),
        );

        metric.update(&input, &MetricMetadata::fake());
        metric.clear();

        assert!(metric.value().unwrap().current().is_nan());
        assert!(metric.running_value().unwrap().current().is_nan());
        assert!(metric.final_value().current().is_nan());
    }

    #[rstest]
    #[case(0.0)]
    #[case(-1.0)]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[should_panic(expected = "epsilon must be finite and positive")]
    fn test_dice_invalid_epsilon(#[case] epsilon: f64) {
        let _ = DiceMetric::<4>::with_config(DiceMetricConfig {
            epsilon,
            ..Default::default()
        });
    }

    #[test]
    fn test_dice_empty_masks() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[0, 0], [0, 0]]]], &device),
            Tensor::from_data([[[[0, 0], [0, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_no_background() {
        let device = Default::default();
        let mut metric = DiceMetric::<4>::new();
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 1, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 1, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_with_background() {
        let device = Default::default();
        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: true,
        };
        let mut metric = DiceMetric::<4>::with_config(config);
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_ignored_background() {
        let device = Default::default();
        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: false,
        };
        let mut metric = DiceMetric::<4>::with_config(config);
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().unwrap().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "DiceInput requires at least 3 dimensions.")]
    fn test_invalid_input_dimensions() {
        let device = Default::default();
        // D = 2, should panic
        let _ = DiceInput::<2>::new(
            Tensor::from_data([[0.0, 0.0]], &device),
            Tensor::from_data([[0.0, 0.0]], &device),
        );
    }

    #[test]
    #[should_panic(
        expected = "Outputs and targets must have the same dimensions. Got [1, 1, 2, 2] and [1, 1, 2, 3]"
    )]
    fn test_mismatched_shape() {
        let device = Default::default();
        // shapes differ
        let _ = DiceInput::<4>::new(
            Tensor::from_data([[[[0.0; 2]; 2]; 1]; 1], &device),
            Tensor::from_data([[[[0.0; 3]; 2]; 1]; 1], &device),
        );
    }

    #[test]
    #[should_panic(expected = "Dice metric requires at least 2 classes when including background.")]
    fn test_include_background_panic() {
        let device = Default::default();
        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: true,
        };
        let mut metric = DiceMetric::<4>::with_config(config);
        let input = DiceInput::new(
            Tensor::from_data([[[[1.0; 2]; 1]; 1]; 1], &device),
            Tensor::from_data([[[[1.0; 2]; 1]; 1]; 1], &device),
        );
        // n_classes = 2, should not panic
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: true,
        };
        let mut metric = DiceMetric::<4>::with_config(config);
        let input = DiceInput::new(
            Tensor::from_data([[[[1.0; 1]; 1]; 1]; 1], &device),
            Tensor::from_data([[[[1.0; 1]; 1]; 1]; 1], &device),
        );
        // n_classes = 1, should panic
        let _entry = metric.update(&input, &MetricMetadata::fake());
    }
}
