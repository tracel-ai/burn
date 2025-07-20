use super::super::{
    Metric, MetricEntry, MetricMetadata, Numeric,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::{ElementConversion, Float, TensorKind},
};
use core::marker::PhantomData;

/// Input type for the [DiceMetric].
///
/// # Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions (default 2).
/// - `K`: Tensor element type (default `Float`).
#[derive(new)]
pub struct DiceInput<B: Backend, const D: usize = 2, K = Float>
where
    K: TensorKind<B>,
{
    /// Model outputs (predictions), as a tensor.
    outputs: Tensor<B, D, K>,
    /// Ground truth targets, as a tensor.
    targets: Tensor<B, D, K>,
}

/// Configuration for the [DiceMetric].
#[derive(Debug, Clone, Copy)]
pub struct DiceMetricConfig {
    /// Epsilon value to avoid division by zero.
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

/// The Dice metric for evaluating overlap between two binary masks.
///
/// # Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions (default 2).
#[derive(Default)]
pub struct DiceMetric<B: Backend, const D: usize = 2> {
    /// Internal state for numeric metric aggregation.
    state: NumericMetricState,
    /// Marker for backend type.
    _b: PhantomData<B>,
    /// Configuration for the metric.
    config: DiceMetricConfig,
}

impl<B: Backend, const D: usize> DiceMetric<B, D> {
    /// Creates a new Dice metric instance with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Dice metric with a custom config.
    pub fn with_config(config: DiceMetricConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }
}

impl<B: Backend, const D: usize> Metric for DiceMetric<B, D> {
    type Input = DiceInput<B, D>;

    fn name(&self) -> String {
        format!("{D}D Dice Metric")
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        // Dice coefficient: 2 * (|X ∩ Y|) / (|X| + |Y|)
        let outputs = item.outputs.clone();
        let targets = item.targets.clone();

        if outputs.dims() != targets.dims() {
            panic!(
                "Outputs and targets must have the same dimensions. Got {:?} and {:?}",
                outputs.dims(),
                targets.dims()
            );
        }

        let dims = outputs.dims();
        let batch_size = dims[0];
        let n_classes = dims[1];
        if self.config.include_background && n_classes < 2 {
            panic!("Dice metric requires at least 2 classes when including background.");
        }

        let intersection = (outputs.clone() * targets.clone()).sum();
        let outputs_sum = outputs.sum();
        let targets_sum = targets.sum();

        // Convert to f64
        let intersection_val = intersection.into_scalar().elem::<f64>();
        let outputs_sum_val = outputs_sum.into_scalar().elem::<f64>();
        let targets_sum_val = targets_sum.into_scalar().elem::<f64>();

        // Use epsilon from config
        let epsilon = self.config.epsilon;
        let dice =
            (2.0 * intersection_val + epsilon) / (outputs_sum_val + targets_sum_val + epsilon);

        self.state.update(
            dice,
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    /// Clears the metric state.
    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend, const D: usize> Numeric for DiceMetric<B, D> {
    /// Returns the current value of the metric.
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_core::tensor::Tensor;

    #[test]
    fn test_dice_perfect_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend>::new();
        let input = DiceInput::new(
            Tensor::from_data([[1.0, 0.0, 1.0, 0.0]], &device),
            Tensor::from_data([[1.0, 0.0, 1.0, 0.0]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_no_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend>::new();
        let input = DiceInput::new(
            Tensor::from_data([[1.0, 0.0, 1.0, 0.0]], &device),
            Tensor::from_data([[0.0, 1.0, 0.0, 1.0]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!(metric.value() < 1e-6);
    }

    #[test]
    fn test_dice_partial_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend>::new();
        let input = DiceInput::new(
            Tensor::from_data([[1.0, 1.0, 0.0, 0.0]], &device),
            Tensor::from_data([[1.0, 0.0, 1.0, 0.0]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        // intersection = 1, sum = 2+2=4, dice = 2*1/4 = 0.5
        assert!((metric.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dice_empty_masks() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend>::new();
        let input = DiceInput::new(
            Tensor::from_data([[0.0, 0.0, 0.0, 0.0]], &device),
            Tensor::from_data([[0.0, 0.0, 0.0, 0.0]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value() - 1.0).abs() < 1e-6);
    }
}
