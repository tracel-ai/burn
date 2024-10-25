use super::{
    classification::{ClassAverageType, ClassificationInput},
    confusion_stats::ConfusionStats,
    state::{FormatOptions, NumericMetricState},
    Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::cast::ToElement,
};
use core::marker::PhantomData;

/// The precision metric.
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    class_average: ClassAverageType,
    threshold: Option<f64>,
    top_k: Option<usize>,
}

#[allow(dead_code)]
impl<B: Backend> PrecisionMetric<B> {
    ///convert to averaged metric, returns float
    fn class_average(&self, mut aggregated_metric: Tensor<B, 1>) -> f64 {
        use ClassAverageType::*;
        let avg_tensor = match self.class_average {
            Micro => aggregated_metric,
            Macro => {
                if aggregated_metric.contains_nan().any().into_scalar() {
                    let nan_mask = aggregated_metric.is_nan();
                    aggregated_metric = aggregated_metric
                        .clone()
                        .select(0, nan_mask.bool_not().argwhere().squeeze(1))
                }
                aggregated_metric.mean()
            }
        };
        avg_tensor.into_scalar().to_f64()
    }

    /// Sets the class average.
    pub fn with_class_average(mut self, class_average: ClassAverageType) -> Self {
        self.class_average = class_average;
        self
    }

    /// Sets the threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self.top_k = None;
        self
    }

    /// Sets the top k.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self.threshold = None;
        self
    }
}

impl<B: Backend> Default for PrecisionMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: PhantomData,
            threshold: Some(0.5),
            class_average: ClassAverageType::Micro,
            top_k: None,
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
        let cf_stats = ConfusionStats::new(
            predictions,
            targets,
            self.threshold,
            self.top_k,
            self.class_average,
        );
        let metric =
            self.class_average(cf_stats.clone().true_positive() / cf_stats.predicted_positive());

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
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, 0.5)]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, 0.5)]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), 3.0/5.0)]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), 4.0/10.0)]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), (0.5 + 0.5 + 1.0)/3.0)]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), (0.5 + 1.0/4.0 + 0.5)/3.0)]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, 5.0/8.0)]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, (2.0/3.0 + 2.0/3.0 + 0.5)/3.0)]
    fn test_precision(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&class_type);
        let mut metric = PrecisionMetric::<TestBackend>::default();
        metric = match class_type {
            Multiclass => metric.with_top_k(top_k.unwrap()),
            _ => metric.with_threshold(threshold.unwrap()),
        };
        metric = metric.with_class_average(avg_type);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
