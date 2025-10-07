use crate::metric::MetricName;

use super::{
    Metric, MetricEntry, MetricMetadata, Numeric,
    classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
    confusion_stats::{ConfusionStats, ConfusionStatsInput},
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::cast::ToElement,
};
use core::marker::PhantomData;
use std::{num::NonZeroUsize, sync::Arc};

///The Recall Metric
#[derive(Clone)]
pub struct RecallMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    _b: PhantomData<B>,
    config: ClassificationMetricConfig,
}

impl<B: Backend> Default for RecallMetric<B> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<B: Backend> RecallMetric<B> {
    fn new(config: ClassificationMetricConfig) -> Self {
        let state = Default::default();
        let name = Arc::new(format!(
            "Recall @ {:?} [{:?}]",
            config.decision_rule, config.class_reduction
        ));

        Self {
            state,
            config,
            name,
            _b: Default::default(),
        }
    }
    /// Recall metric for binary classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn binary(threshold: f64) -> Self {
        Self::new(ClassificationMetricConfig {
            decision_rule: DecisionRule::Threshold(threshold),
            // binary classification results are the same independently of class_reduction
            ..Default::default()
        })
    }

    /// Recall metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multiclass(top_k: usize, class_reduction: ClassReduction) -> Self {
        Self::new(ClassificationMetricConfig {
            decision_rule: DecisionRule::TopK(
                NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
            ),
            class_reduction,
        })
    }

    /// Recall metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multilabel(threshold: f64, class_reduction: ClassReduction) -> Self {
        Self::new(ClassificationMetricConfig {
            decision_rule: DecisionRule::Threshold(threshold),
            class_reduction,
        })
    }

    fn class_average(&self, mut aggregated_metric: Tensor<B, 1>) -> f64 {
        use ClassReduction::{Macro, Micro};
        let avg_tensor = match self.config.class_reduction {
            Micro => aggregated_metric,
            Macro => {
                if aggregated_metric
                    .clone()
                    .contains_nan()
                    .any()
                    .into_scalar()
                    .to_bool()
                {
                    let nan_mask = aggregated_metric.clone().is_nan();
                    aggregated_metric = aggregated_metric
                        .clone()
                        .select(0, nan_mask.bool_not().argwhere().squeeze_dim(1))
                }
                aggregated_metric.mean()
            }
        };
        avg_tensor.into_scalar().to_f64()
    }
}

impl<B: Backend> Metric for RecallMetric<B> {
    type Input = ConfusionStatsInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [sample_size, _] = input.predictions.dims();

        let cf_stats = ConfusionStats::new(input, &self.config);
        let metric = self.class_average(cf_stats.clone().true_positive() / cf_stats.positive());

        self.state.update(
            100.0 * metric,
            sample_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }
}

impl<B: Backend> Numeric for RecallMetric<B> {
    fn value(&self) -> super::NumericEntry {
        self.state.value()
    }
    fn attributes(&self) -> super::NumericAttributes {
        super::NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassReduction::{self, *},
        Metric, MetricMetadata, Numeric, RecallMetric,
    };
    use crate::{
        TestBackend,
        tests::{ClassificationType, THRESHOLD, dummy_classification_input},
    };
    use burn_core::tensor::{TensorData, Tolerance};
    use rstest::rstest;

    #[rstest]
    #[case::binary(THRESHOLD, 0.5)]
    fn test_binary_recall(#[case] threshold: f64, #[case] expected: f64) {
        let input = dummy_classification_input(&ClassificationType::Binary).into();
        let mut metric = RecallMetric::binary(threshold);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default())
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
        TensorData::from([metric.value().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default())
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
        TensorData::from([metric.value().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default())
    }

    #[test]
    fn test_parameterized_unique_name() {
        let metric_a = RecallMetric::<TestBackend>::multiclass(1, ClassReduction::Macro);
        let metric_b = RecallMetric::<TestBackend>::multiclass(2, ClassReduction::Macro);
        let metric_c = RecallMetric::<TestBackend>::multiclass(1, ClassReduction::Macro);

        assert_ne!(metric_a.name(), metric_b.name());
        assert_eq!(metric_a.name(), metric_c.name());

        let metric_a = RecallMetric::<TestBackend>::binary(0.5);
        let metric_b = RecallMetric::<TestBackend>::binary(0.75);
        assert_ne!(metric_a.name(), metric_b.name());
    }
}
