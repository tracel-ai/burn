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

#[derive(new)]
/// The precision metric.
pub struct PrecisionMetric<B: Backend> {
    #[new(default)]
    state: NumericMetricState,
    _b: PhantomData<B>,
    #[new(default)]
    class_reduction: ClassReduction,
    threshold: Option<f64>,
    top_k: Option<usize>,
}

impl<B: Backend> PrecisionMetric<B> {
    ///convert to averaged metric, returns float
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

    pub fn with_class_reduction(mut self, class_reduction: ClassReduction) -> Self {
        self.class_reduction = class_reduction;
        self
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
            self.top_k.map(NonZeroUsize::new).flatten(),
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
        ClassReduction::{self, *},
        Metric, MetricMetadata, Numeric, PrecisionMetric,
    };
    use crate::tests::{
        dummy_classification_input,
        ClassificationType::{self, *},
        THRESHOLD,
    };
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
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&classification_type);
        let mut metric = PrecisionMetric::new(
            threshold,
            top_k
        );
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
