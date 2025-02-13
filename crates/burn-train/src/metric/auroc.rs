use core::f64;
use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The Area Under the Receiver Operating Characteristic Curve (AUROC, also referred to as [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)) for binary classification.
#[derive(Default)]
pub struct AurocMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

/// The [AUROC metric](AurocMetric) input type.
#[derive(new)]
pub struct AurocInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AurocMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    fn binary_auroc(&self, probabilities: &Tensor<B, 1>, targets: &Tensor<B, 1, Int>) -> f64 {
        let n = targets.dims()[0];

        let n_pos = targets.clone().sum().into_scalar().elem::<u64>() as usize;

        // Early return if we don't have both positive and negative samples
        if n_pos == 0 || n_pos == n {
            if n_pos == 0 {
                log::warn!("Metric cannot be computed because all target values are negative.")
            } else {
                log::warn!("Metric cannot be computed because all target values are positive.")
            }
            return 0.0;
        }

        let pos_mask = targets.clone().equal_elem(1).int().reshape([n, 1]);
        let neg_mask = targets.clone().equal_elem(0).int().reshape([1, n]);

        let valid_pairs = pos_mask * neg_mask;

        let prob_i = probabilities.clone().reshape([n, 1]).repeat_dim(1, n);
        let prob_j = probabilities.clone().reshape([1, n]).repeat_dim(0, n);

        let correct_order = prob_i.clone().greater(prob_j.clone()).int();

        let ties = prob_i.equal(prob_j).int();

        // Calculate AUC components
        let num_pairs = valid_pairs.clone().sum().into_scalar().elem::<f64>();
        let correct_pairs = (correct_order * valid_pairs.clone())
            .sum()
            .into_scalar()
            .elem::<f64>();
        let tied_pairs = (ties * valid_pairs).sum().into_scalar().elem::<f64>();

        (correct_pairs + 0.5 * tied_pairs) / num_pairs
    }
}

impl<B: Backend> Metric for AurocMetric<B> {
    type Input = AurocInput<B>;

    fn update(&mut self, input: &AurocInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, num_classes] = input.outputs.dims();

        assert_eq!(
            num_classes, 2,
            "Currently only binary classification is supported"
        );

        let probabilities = {
            let exponents = input.outputs.clone().exp();
            let sum = exponents.clone().sum_dim(1);
            (exponents / sum)
                .select(1, Tensor::arange(1..2, &input.outputs.device()))
                .squeeze(1)
        };

        let area_under_curve = self.binary_auroc(&probabilities, &input.targets);

        self.state.update(
            100.0 * area_under_curve,
            batch_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> String {
        "AUROC".to_string()
    }
}

impl<B: Backend> Numeric for AurocMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_auroc() {
        let device = Default::default();
        let mut metric = AurocMetric::<TestBackend>::new();

        let input = AurocInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.9], // High confidence positive
                    [0.7, 0.3], // Low confidence negative
                    [0.6, 0.4], // Low confidence negative
                    [0.2, 0.8], // High confidence positive
                ],
                &device,
            ),
            Tensor::from_data([1, 0, 0, 1], &device), // True labels
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 100.0);
    }

    #[test]
    fn test_auroc_perfect_separation() {
        let device = Default::default();
        let mut metric = AurocMetric::<TestBackend>::new();

        let input = AurocInput::new(
            Tensor::from_data([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device),
            Tensor::from_data([1, 0, 0, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 100.0); // Perfect AUC
    }

    #[test]
    fn test_auroc_random() {
        let device = Default::default();
        let mut metric = AurocMetric::<TestBackend>::new();

        let input = AurocInput::new(
            Tensor::from_data(
                [
                    [0.5, 0.5], // Random predictions
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
                &device,
            ),
            Tensor::from_data([1, 0, 0, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 50.0);
    }

    #[test]
    fn test_auroc_all_one_class() {
        let device = Default::default();
        let mut metric = AurocMetric::<TestBackend>::new();

        let input = AurocInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.9], // All positives predictions
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6],
                ],
                &device,
            ),
            Tensor::from_data([1, 1, 1, 1], &device), // All positive class
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(metric.value(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Currently only binary classification is supported")]
    fn test_auroc_multiclass_error() {
        let device = Default::default();
        let mut metric = AurocMetric::<TestBackend>::new();

        let input = AurocInput::new(
            Tensor::from_data(
                [
                    [0.1, 0.2, 0.7], // More than 2 classes not supported
                    [0.3, 0.5, 0.2],
                ],
                &device,
            ),
            Tensor::from_data([2, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
    }
}
