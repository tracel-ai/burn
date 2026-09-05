use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, SerializedEntry};
use burn_core::tensor::{Int, Tensor};

/// The accuracy metric.
#[derive(Clone)]
pub struct AccuracyMetric {
    name: MetricName,
    state: NumericMetricState,
    pad_token: Option<usize>,
}

/// The [accuracy metric](AccuracyMetric) input type.
#[derive(new)]
pub struct AccuracyInput {
    outputs: Tensor<2>,
    targets: Tensor<1, Int>,
}

impl Default for AccuracyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl AccuracyMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("Accuracy".to_string()),
            state: Default::default(),
            pad_token: Default::default(),
        }
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl Metric for AccuracyMetric {
    type Input = AccuracyInput;

    fn update(&mut self, input: &AccuracyInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [batch_size, _n_classes] = outputs.dims();

        let outputs = outputs.argmax(1).reshape([batch_size]);

        let (num_matches, num_pad) = match self.pad_token {
            Some(pad_token) => {
                let mask = targets.clone().equal_scalar(pad_token as i64);
                let matches = outputs.equal(targets).float().mask_fill(mask.clone(), 0);
                let num_pad = mask.int().sum().into_scalar::<i64>() as usize;

                (matches.sum().into_scalar::<f64>(), num_pad)
            }
            None => (outputs.equal(targets).int().sum().into_scalar::<f64>(), 0),
        };
        let valid_count = batch_size - num_pad;
        let accuracy = num_matches / valid_count as f64;

        self.state.update(100.0 * accuracy, valid_count);
        self.state
            .compute_update(FormatOptions::new(self.name()).unit("%").precision(2))
    }

    fn compute(&mut self) -> SerializedEntry {
        self.state
            .compute_final(FormatOptions::new(self.name()).unit("%").precision(2))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        super::NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for AccuracyMetric {
    fn value(&self) -> Option<super::NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<super::NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> super::NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ClassificationOutput,
        metric::{Adaptor, ItemLazy},
    };
    use burn_core::tensor::Device;

    #[test]
    fn test_accuracy_without_padding() {
        let device = Default::default();
        let mut metric = AccuracyMetric::new();
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
        assert_eq!(50.0, metric.value().unwrap().current());
    }

    #[test]
    fn test_accuracy_with_padding() {
        let device = Default::default();
        let mut metric = AccuracyMetric::new().with_pad_token(3);
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
        assert_eq!(50.0, metric.value().unwrap().current());
    }

    #[test]
    fn test_accuracy_epoch_aggregation_excludes_padding() {
        let device = Default::default();
        let mut metric = AccuracyMetric::new().with_pad_token(2);

        // One valid, correct sample and three padding samples.
        metric.update(
            &AccuracyInput::new(
                Tensor::from_data([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]], &device),
                Tensor::from_data([0, 2, 2, 2], &device),
            ),
            &MetricMetadata::fake(),
        );
        // Four valid, incorrect samples.
        metric.update(
            &AccuracyInput::new(
                Tensor::from_data([[0.9, 0.1]; 4], &device),
                Tensor::from_data([1, 1, 1, 1], &device),
            ),
            &MetricMetadata::fake(),
        );

        // One correct prediction out of five valid samples.
        assert_eq!(20.0, metric.final_value().current());
    }

    #[test]
    fn test_fully_padded_batch_does_not_poison_epoch_accuracy() {
        let device = Default::default();
        let mut metric = AccuracyMetric::new().with_pad_token(2);

        metric.update(
            &AccuracyInput::new(
                Tensor::from_data([[0.9, 0.1]; 2], &device),
                Tensor::from_data([2, 2], &device),
            ),
            &MetricMetadata::fake(),
        );
        metric.update(
            &AccuracyInput::new(
                Tensor::from_data([[0.9, 0.1]], &device),
                Tensor::from_data([0], &device),
            ),
            &MetricMetadata::fake(),
        );

        assert_eq!(100.0, metric.final_value().current());
    }

    #[test]
    fn test_accuracy_after_syncing_autodiff_classification_output() {
        let device = Device::flex().autodiff();
        let mut metric = AccuracyMetric::new();
        let output = ClassificationOutput::new(
            Tensor::from_data([0.0], &device),
            Tensor::from_data([[0.1, 0.9]], &device),
            Tensor::from_data([1], &device),
        )
        .sync();
        let input = output.adapt();

        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_eq!(100.0, metric.value().unwrap().current());
    }
}
