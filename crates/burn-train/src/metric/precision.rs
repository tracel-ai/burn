use super::{
    confusion_matrix::ConfusionMatrix,
    state::{FormatOptions, NumericMetricState},
    ClassificationAverage, ClassificationInput, ClassificationMetric, Metric, MetricEntry,
    MetricMetadata, Numeric,
};
use burn_core::tensor::backend::Backend;
use core::marker::PhantomData;

/// The precision metric.
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    threshold: f64,
    average: ClassificationAverage,
}

impl<B: Backend> Default for PrecisionMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: PhantomData,
            threshold: 0.5,
            average: ClassificationAverage::Micro,
        }
    }
}

impl<B: Backend> Metric for PrecisionMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;
    fn update(
        &mut self,
        input: &ClassificationInput<B>,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let [sample_size, _] = input.predictions.dims();

        let conf_mat = ConfusionMatrix::new(input, self.threshold, self.average);
        let agg_precision = conf_mat.clone().true_positive() / conf_mat.predicted_positive();
        let precision = self.average.to_averaged_metric(agg_precision);

        self.state.update(
            100.0 * precision,
            sample_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for PrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

impl<B: Backend> ClassificationMetric<B> for PrecisionMetric<B> {
    fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    fn with_average(mut self, average: ClassificationAverage) -> Self {
        self.average = average;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassificationAverage, Metric, MetricMetadata, Numeric, PrecisionMetric};
    use crate::metric::ClassificationMetric;
    use crate::tests::{dummy_classification_input, ClassificationType, THRESHOLD};
    use crate::TestBackend;
    use approx::assert_relative_eq;
    use strum::IntoEnumIterator;

    #[test]
    fn test_precision() {
        for class_avg_type in ClassificationAverage::iter() {
            for classification_type in ClassificationType::iter() {
                let (input, target_diff) = dummy_classification_input(&classification_type);
                let mut metric = PrecisionMetric::<TestBackend>::default()
                    .with_threshold(THRESHOLD)
                    .with_average(class_avg_type);
                let _entry = metric.update(&input, &MetricMetadata::fake());

                //tp/(tp+fp) = 1 - fp/(tp+fp)
                let metric_precision = metric.value();

                //fp/(tp+fp+tn+fn) = fp/(tp+fp)(1 + negative/positive)
                let agg_false_positive_rate =
                    class_avg_type.aggregate_mean(target_diff.clone().equal_elem(-1));
                let pred_positive = input.targets.clone().int() - target_diff.clone().int();
                let agg_pred_negative =
                    class_avg_type.aggregate_sum(pred_positive.clone().bool().bool_not());
                let agg_pred_positive = class_avg_type.aggregate_sum(pred_positive.bool());
                //1 - fp(1 + negative/positive)/(tp+fp+tn+fn) = 1 - fp/(tp+fp) = tp/(tp+fp)
                let test_precision = class_avg_type.to_averaged_metric(
                    -agg_false_positive_rate * (agg_pred_negative / agg_pred_positive + 1.0) + 1.0,
                );
                assert_relative_eq!(
                    metric_precision,
                    test_precision * 100.0,
                    max_relative = 1e-3
                );
            }
        }
    }
}
