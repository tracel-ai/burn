use super::{
    classification::{ClassAverageType, ClassificationInput},
    confusion_stats::ConfusionStats,
    state::{FormatOptions, NumericMetricState},
    Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn_core::tensor::backend::Backend;
use core::marker::PhantomData;

/// The precision metric.
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    threshold: f64,
    class_average: ClassAverageType,
}
#[allow(dead_code)]
impl<B: Backend> PrecisionMetric<B> {
    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the class average.
    pub fn with_class_average(mut self, class_average: ClassAverageType) -> Self {
        self.class_average = class_average;
        self
    }
}

impl<B: Backend> Default for PrecisionMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: PhantomData,
            threshold: 0.5,
            class_average: ClassAverageType::Micro,
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
        let (predictions, targets) = input.clone().into();
        let [sample_size, _] = input.predictions.dims();
        let metric = ConfusionStats::new(predictions, targets, self.threshold, self.class_average)
            .precision();

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

#[cfg(test)]
mod tests {
    use super::{
        ClassAverageType::{self, *},
        Metric, MetricMetadata, Numeric, PrecisionMetric,
    };
    use crate::tests::{
        dummy_classification_input,
        ClassificationType::{self, *},
        THRESHOLD,
    };
    use crate::TestBackend;
    use burn_core::tensor::TensorData;
    use rstest::rstest;

    #[rstest]
    #[case::binary_micro(Binary, Micro, 0.5)]
    #[case::binary_macro(Binary, Macro, 0.5)]
    #[case::multiclass_micro(Multiclass, Micro, 3.0/5.0)]
    #[case::multiclass_macro(Multiclass, Macro, (0.5 + 0.5 + 1.0)/3.0)]
    #[case::multilabel_micro(Multilabel, Micro, 5.0/8.0)]
    #[case::multilabel_macro(Multilabel, Macro, (2.0/3.0 + 2.0/3.0 + 0.5)/3.0)]
    fn test_precision(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&class_type);
        let mut metric = PrecisionMetric::<TestBackend>::default()
            .with_threshold(THRESHOLD)
            .with_class_average(avg_type);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
