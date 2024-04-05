use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

/// The accuracy metric.
#[derive(Default)]
pub struct AccuracyMetric<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [accuracy metric](AccuracyMetric) input type.
#[derive(new)]
pub struct AccuracyInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> AccuracyMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl<B: Backend> Metric for AccuracyMetric<B> {
    const NAME: &'static str = "Accuracy";

    type Input = AccuracyInput<B>;

    fn update(&mut self, input: &AccuracyInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _n_classes] = input.outputs.dims();

        let targets = input.targets.clone().to_device(&B::Device::default());
        let outputs = input
            .outputs
            .clone()
            .argmax(1)
            .to_device(&B::Device::default())
            .reshape([batch_size]);

        let accuracy = match self.pad_token {
            Some(pad_token) => {
                let mask = targets.clone().equal_elem(pad_token as i64);
                let matches = outputs.equal(targets).int().mask_fill(mask.clone(), 0);
                let num_pad = mask.int().sum().into_scalar().elem::<f64>();

                matches.sum().into_scalar().elem::<f64>() / (batch_size as f64 - num_pad)
            }
            None => {
                outputs
                    .equal(targets)
                    .int()
                    .sum()
                    .into_scalar()
                    .elem::<f64>()
                    / batch_size as f64
            }
        };

        self.state.update(
            100.0 * accuracy,
            batch_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for AccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_accuracy_without_padding() {
        let device = Default::default();
        let mut metric = AccuracyMetric::<TestBackend>::new();
        let input = AccuracyInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8], // 2
                    [1.0, 2.0, 0.5], // 1
                    [0.4, 0.1, 0.2], // 0
                    [0.6, 0.7, 0.2], // 1
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    #[test]
    fn test_accuracy_with_padding() {
        let device = Default::default();
        let mut metric = AccuracyMetric::<TestBackend>::new().with_pad_token(3);
        let input = AccuracyInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.2, 0.8, 0.0], // 2
                    [1.0, 2.0, 0.5, 0.0], // 1
                    [0.4, 0.1, 0.2, 0.0], // 0
                    [0.6, 0.7, 0.2, 0.0], // 1
                    [0.0, 0.1, 0.2, 5.0], // Predicted padding should not count
                    [0.0, 0.1, 0.2, 0.0], // Error on padding should not count
                    [0.6, 0.0, 0.2, 0.0], // Error on padding should not count
                ],
                &device,
            ),
            Tensor::from_data([2, 2, 1, 1, 3, 3, 3], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }
}
