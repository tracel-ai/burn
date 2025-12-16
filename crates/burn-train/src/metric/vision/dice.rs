use crate::metric::{MetricAttributes, MetricName, SerializedEntry};

use super::super::{
    Metric, MetricMetadata,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::{ElementConversion, Int, s},
};
use core::marker::PhantomData;

/// Input type for the [DiceMetric].
///
/// # Type Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions. Should be more than, or equal to 3 (default 4).
pub struct DiceInput<B: Backend, const D: usize = 4> {
    /// Model outputs (predictions), as a tensor.
    outputs: Tensor<B, D, Int>,
    /// Ground truth targets, as a tensor.
    targets: Tensor<B, D, Int>,
}

impl<B: Backend, const D: usize> DiceInput<B, D> {
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
    pub fn new(outputs: Tensor<B, D, Int>, targets: Tensor<B, D, Int>) -> Self {
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

/// The Dice-Sorenson coefficient (DSC) for evaluating overlap between two binary masks.
/// The DSC is defined as:
/// `DSC = 2 * (|X ∩ Y|) / (|X| + |Y|)`
/// where `X` is the model output and `Y` is the ground truth target.
///
///  # Type Parameters
/// - `B`: Backend type.
/// - `D`: Number of dimensions. Should be more than, or equal to 3 (default 4).
#[derive(Default, Clone)]
pub struct DiceMetric<B: Backend, const D: usize = 4> {
    name: MetricName,
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
        Self::with_config(DiceMetricConfig::default())
    }

    /// Creates a new Dice metric with a custom config.
    pub fn with_config(config: DiceMetricConfig) -> Self {
        let name = MetricName::new(format!("{D}D Dice Metric"));
        assert!(D >= 3, "DiceMetric requires at least 3 dimensions.");
        Self {
            name,
            config,
            ..Default::default()
        }
    }
}

impl<B: Backend, const D: usize> Metric for DiceMetric<B, D> {
    type Input = DiceInput<B, D>;

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

    fn attributes(&self) -> MetricAttributes {
        crate::metric::NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend, const D: usize> crate::metric::Numeric for DiceMetric<B, D> {
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
    use burn_core::tensor::{Shape, Tensor};

    #[test]
    fn test_dice_perfect_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend, 4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_no_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend, 4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
            Tensor::from_data([[[[0, 1], [0, 1]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!(metric.value().current() < 1e-6);
    }

    #[test]
    fn test_dice_partial_overlap() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend, 4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[1, 1], [0, 0]]]], &device),
            Tensor::from_data([[[[1, 0], [1, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        // intersection = 1, sum = 2+2=4, dice = 2*1/4 = 0.5
        assert!((metric.value().current() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dice_empty_masks() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend, 4>::new();
        let input = DiceInput::new(
            Tensor::from_data([[[[0, 0], [0, 0]]]], &device),
            Tensor::from_data([[[[0, 0], [0, 0]]]], &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_no_background() {
        let device = Default::default();
        let mut metric = DiceMetric::<TestBackend, 4>::new();
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 1, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 1, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_with_background() {
        let device = Default::default();
        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: true,
        };
        let mut metric = DiceMetric::<TestBackend, 4>::with_config(config);
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_ignored_background() {
        let device = Default::default();
        let config = DiceMetricConfig {
            epsilon: 1e-7,
            include_background: false,
        };
        let mut metric = DiceMetric::<TestBackend, 4>::with_config(config);
        let input = DiceInput::new(
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
            Tensor::ones(Shape::new([1, 2, 2, 2]), &device),
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert!((metric.value().current() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "DiceInput requires at least 3 dimensions.")]
    fn test_invalid_input_dimensions() {
        let device = Default::default();
        // D = 2, should panic
        let _ = DiceInput::<TestBackend, 2>::new(
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
        let _ = DiceInput::<TestBackend, 4>::new(
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
        let mut metric = DiceMetric::<TestBackend, 4>::with_config(config);
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
        let mut metric = DiceMetric::<TestBackend, 4>::with_config(config);
        let input = DiceInput::new(
            Tensor::from_data([[[[1.0; 1]; 1]; 1]; 1], &device),
            Tensor::from_data([[[[1.0; 1]; 1]; 1]; 1], &device),
        );
        // n_classes = 1, should panic
        let _entry = metric.update(&input, &MetricMetadata::fake());
    }
}
