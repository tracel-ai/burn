use super::{
    classification::{ClassReduction, ClassificationDecisionRule, ClassificationMetricConfig},
    confusion_stats::{ConfusionStats, ConfusionStatsInput},
    state::{FormatOptions, NumericMetricState},
    Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::cast::ToElement,
};
use core::marker::PhantomData;
use std::num::NonZeroUsize;

///The Precision Metric
#[derive(Default)]
pub struct RecallMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    config: ClassificationMetricConfig,
}

impl<B: Backend> RecallMetric<B> {
    /// Recall metric for binary classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn binary(threshold: f64) -> Self {
        Self {
            config: ClassificationMetricConfig {
                decision_rule: ClassificationDecisionRule::Threshold(threshold),
                // binary classification results are the same independently of class_reduction
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Recall metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
    #[allow(dead_code)]
    pub fn multiclass(top_k: usize, class_reduction: ClassReduction) -> Self {
        Self {
            config: ClassificationMetricConfig {
                decision_rule: ClassificationDecisionRule::TopK(
                    NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
                ),
                class_reduction,
            },
            ..Default::default()
        }
    }

    /// Recall metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn multilabel(threshold: f64, class_reduction: ClassReduction) -> Self {
        Self {
            config: ClassificationMetricConfig {
                decision_rule: ClassificationDecisionRule::Threshold(threshold),
                class_reduction,
            },
            ..Default::default()
        }
    }

    fn class_average(&self, mut aggregated_metric: Tensor<B, 1>) -> f64 {
        use ClassReduction::{Macro, Micro};
        let avg_tensor = match self.config.class_reduction {
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

impl<B: Backend> Metric for RecallMetric<B> {
    const NAME: &'static str = "Recall";
    type Input = ConfusionStatsInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [sample_size, _] = input.predictions.dims();

        let cf_stats = ConfusionStats::new(input, &self.config);
        let metric = self.class_average(cf_stats.clone().true_positive() / cf_stats.positive());

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

impl<B: Backend> Numeric for RecallMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassReduction::{self, *},
        Metric, MetricMetadata, Numeric, RecallMetric,
    };
    use crate::tests::{dummy_classification_input, ClassificationType, THRESHOLD};
    use burn_core::tensor::TensorData;
    use rstest::rstest;

    #[rstest]
    #[case::binary_macro(THRESHOLD, 0.5)]
    fn test_binary_recall(#[case] threshold: f64, #[case] expected: f64) {
        let input = dummy_classification_input(&ClassificationType::Binary).into();
        let mut metric = RecallMetric::binary(threshold);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }

    #[rstest]
    #[case::multiclass_micro_k1(Micro, 1, 3.0/5.0)]
    #[case::multiclass_micro_k2(Micro, 2, 4.0/5.0)]
    #[case::multiclass_macro_k1(Macro, 1, (0.5 + 1.0 + 0.5)/3.0)]
    #[case::multiclass_macro_k2(Macro, 2, (1.0 + 1.0 + 0.5)/3.0)]
    fn test_multiclass_recall(
        #[case] class_reduction: ClassReduction,
        #[case] top_k: usize,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multiclass).into();
        let mut metric = RecallMetric::multiclass(top_k, class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }

    #[rstest]
    #[case::multilabel_micro(Micro, THRESHOLD, 5.0/9.0)]
    #[case::multilabel_macro(Macro, THRESHOLD, (0.5 + 1.0 + 1.0/3.0)/3.0)]
    fn test_multilabel_recall(
        #[case] class_reduction: ClassReduction,
        #[case] threshold: f64,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multilabel).into();
        let mut metric = RecallMetric::multilabel(threshold, class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
