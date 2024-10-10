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
        let agg_metric = conf_mat.clone().true_positive() / conf_mat.predicted_positive();
        let metric = self.average.to_averaged_metric(agg_metric);

        self.state.update(
            100.0 * metric,
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
    use super::{
        ClassificationAverage::{self, *},
        Metric, MetricMetadata, Numeric, PrecisionMetric,
    };
    use crate::TestBackend;
    use crate::{
        metric::ClassificationMetric,
        tests::{
            dummy_classification_input,
            ClassificationType::{self, *},
            THRESHOLD,
        },
    };
    use approx::assert_relative_eq;
    use yare::parameterized;

    #[parameterized(
    binary_micro = {Binary, Micro, 0.5},
    binary_macro = {Binary, Macro, 0.5},
    multiclass_micro = {Multiclass, Micro, 3.0/5.0},
    multiclass_macro = {Multiclass, Macro, (0.5 + 0.5 + 1.0)/3.0},
    multilabel_micro = {Multilabel, Micro, 5.0/8.0},
    multilabel_macro = {Multilabel, Macro, (2.0/3.0 + 2.0/3.0 + 0.5)/3.0})]
    fn test_presision(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: f64,
    ) {
        let input = dummy_classification_input(&class_type);
        let mut metric = PrecisionMetric::<TestBackend>::default()
            .with_threshold(THRESHOLD)
            .with_average(avg_type);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        assert_relative_eq!(metric.value(), expected * 100.0, max_relative = 1e-3)
    }
}
