use super::{
    classification::{ClassReduction, ClassificationInput},
    confusion_stats::ConfusionStats,
    state::{FormatOptions, NumericMetricState},
    Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::cast::ToElement,
};
use core::marker::PhantomData;
use std::num::NonZeroUsize;

/// PrecisionMetric for binary classification task
pub struct BinaryPrecisionMetric<B: Backend>(PrecisionMetric<B>);
impl<B: Backend> BinaryPrecisionMetric<B> {
    /// Initialize PrecisionMetric for binary classification task.
    /// Uses default threshold @ 0.5
    pub fn new() -> Self {
        Default::default()
    }

    /// Set class reduction.
    pub fn with_class_reduction(self, reduction: ClassReduction) -> Self {
        Self(self.0.with_class_reduction(reduction))
    }

    /// Set threshold.
    pub fn with_threshold(self, threshold: f64) -> Self {
        Self(self.0.with_threshold(threshold))
    }
}

impl<B: Backend> Default for BinaryPrecisionMetric<B> {
    fn default() -> Self {
        Self(Default::default()).with_threshold(0.5)
    }
}

impl<B: Backend> Metric for BinaryPrecisionMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        self.0.update(item, metadata)
    }

    fn clear(&mut self) {
        self.0.clear()
    }
}

impl<B: Backend> Numeric for BinaryPrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.0.value()
    }
}

/// PrecisionMetric for multiclass classification task
pub struct MulticlassPrecisionMetric<B: Backend>(PrecisionMetric<B>);
impl<B: Backend> MulticlassPrecisionMetric<B> {
    /// Initialize PrecisionMetric for multiclass classification task
    /// Uses default top_k @ 1 and Macro average
    pub fn new() -> Self {
        Default::default()
    }

    /// Set class reduction.
    pub fn with_class_reduction(self, reduction: ClassReduction) -> Self {
        Self(self.0.with_class_reduction(reduction))
    }

    /// Set top_k.
    pub fn with_top_k(self, top_k: NonZeroUsize) -> Self {
        Self(self.0.with_top_k(top_k))
    }
}

impl<B: Backend> Default for MulticlassPrecisionMetric<B> {
    fn default() -> Self {
        Self(Default::default()).with_top_k(NonZeroUsize::new(1).unwrap())
    }
}

impl<B: Backend> Metric for MulticlassPrecisionMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        self.0.update(item, metadata)
    }

    fn clear(&mut self) {
        self.0.clear()
    }
}

impl<B: Backend> Numeric for MulticlassPrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.0.value()
    }
}

/// PrecisionMetric for multilabel classification task
pub struct MultilabelPrecisionMetric<B: Backend>(PrecisionMetric<B>);
impl<B: Backend> MultilabelPrecisionMetric<B> {
    /// Initialize PrecisionMetric for multilabel classification task
    /// Uses default threshold @ 0.5
    pub fn new() -> Self {
        Default::default()
    }

    /// Set class reduction.
    pub fn with_class_reduction(self, reduction: ClassReduction) -> Self {
        Self(self.0.with_class_reduction(reduction))
    }

    /// Set threshold.
    pub fn with_threshold(self, threshold: f64) -> Self {
        Self(self.0.with_threshold(threshold))
    }
}

impl<B: Backend> Default for MultilabelPrecisionMetric<B> {
    fn default() -> Self {
        Self(Default::default()).with_threshold(0.5)
    }
}

impl<B: Backend> Metric for MultilabelPrecisionMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        self.0.update(item, metadata)
    }

    fn clear(&mut self) {
        self.0.clear()
    }
}

impl<B: Backend> Numeric for MultilabelPrecisionMetric<B> {
    fn value(&self) -> f64 {
        self.0.value()
    }
}

#[derive(Default)]
struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    class_reduction: ClassReduction,
    threshold: Option<f64>,
    top_k: Option<NonZeroUsize>,
}

impl<B: Backend> PrecisionMetric<B> {
    fn with_class_reduction(mut self, class_reduction: ClassReduction) -> Self {
        self.class_reduction = class_reduction;
        self
    }

    fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    fn with_top_k(mut self, top_k: NonZeroUsize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    fn class_average(&self, mut aggregated_metric: Tensor<B, 1>) -> f64 {
        use ClassReduction::*;
        let avg_tensor = match self.class_reduction {
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
}

impl<B: Backend> Metric for PrecisionMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let (predictions, targets) = input.clone().into();
        let [sample_size, _] = input.predictions.dims();
        let cf_stats = ConfusionStats::new(
            predictions,
            targets,
            self.threshold,
            self.top_k,
            self.class_reduction,
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
        BinaryPrecisionMetric,
        ClassReduction::{self, *},
        Metric, MetricMetadata, MulticlassPrecisionMetric, MultilabelPrecisionMetric, Numeric,
    };
    use crate::tests::{dummy_classification_input, ClassificationType, THRESHOLD};
    use burn_core::tensor::TensorData;
    use rstest::rstest;
    use std::num::NonZeroUsize;

    #[rstest]
    #[case::binary_micro(Micro, THRESHOLD, 0.5)]
    #[case::binary_macro(Macro, THRESHOLD, 0.5)]
    fn test_binary_precision(
        #[case] class_reduction: ClassReduction,
        #[case] threshold: f64,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Binary);
        let mut metric = BinaryPrecisionMetric::new()
            .with_threshold(threshold)
            .with_class_reduction(class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }

    #[rstest]
    #[case::multiclass_micro_k1(Micro, 1, 3.0/5.0)]
    #[case::multiclass_micro_k2(Micro, 2, 4.0/10.0)]
    #[case::multiclass_macro_k1(Macro, 1, (0.5 + 0.5 + 1.0)/3.0)]
    #[case::multiclass_macro_k2(Macro, 2, (0.5 + 1.0/4.0 + 0.5)/3.0)]
    fn test_multiclass_precision(
        #[case] class_reduction: ClassReduction,
        #[case] top_k: usize,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multiclass);
        let mut metric = MulticlassPrecisionMetric::new()
            .with_top_k(NonZeroUsize::new(top_k).unwrap())
            .with_class_reduction(class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }

    #[rstest]
    #[case::multilabel_micro(Micro, THRESHOLD, 5.0/8.0)]
    #[case::multilabel_macro(Macro, THRESHOLD, (2.0/3.0 + 2.0/3.0 + 0.5)/3.0)]
    fn test_precision(
        #[case] class_reduction: ClassReduction,
        #[case] threshold: f64,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multilabel);
        let mut metric = MultilabelPrecisionMetric::new()
            .with_threshold(threshold)
            .with_class_reduction(class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
