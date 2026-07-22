use crate::metric::{MetricName, Numeric, state::ConfusionStatsState};

use super::{
    Metric, MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry, SerializedEntry,
    classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
    confusion_stats::{ConfusionStats, ConfusionStatsInput},
    state::FormatOptions,
};
use std::{num::NonZeroUsize, sync::Arc};

/// The Recall Metric
#[derive(Clone)]
pub struct RecallMetric {
    name: MetricName,
    state: ConfusionStatsState,
    config: ClassificationMetricConfig,
}

impl Default for RecallMetric {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl RecallMetric {
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
}

impl Metric for RecallMetric {
    type Input = ConfusionStatsInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [sample_size, _] = input.predictions.dims();

        let stats = ConfusionStats::new(input, &self.config);
        let tp = Some(stats.clone().true_positive());
        let fn_ = Some(stats.false_negative());

        self.state.update(tp, None, fn_, sample_size);
        self.state.compute_update(
            self.config.class_reduction,
            FormatOptions::new(self.name()).unit("%").precision(2),
            |tp, _, fn_| {
                let (tp, fn_) = (tp.unwrap(), fn_.unwrap());
                let denominator = tp.clone() + fn_;
                // Avoid division by zero on empty classes
                let mask = denominator.clone().equal_elem(0.0);
                let actual_positive = denominator.mask_fill(mask, 1.0);

                (tp / actual_positive) * 100.0
            },
        )
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
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for RecallMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.state.current_value()
    }

    fn running_value(&self) -> Option<NumericEntry> {
        self.state.running_value()
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassReduction::{self, *},
        Metric, MetricMetadata, RecallMetric,
    };
    use crate::metric::{ConfusionStatsInput, Numeric};
    use crate::tests::{ClassificationType, THRESHOLD, dummy_classification_input};
    use burn_core::{
        Tensor,
        tensor::{TensorData, Tolerance},
    };
    use rstest::rstest;

    #[rstest]
    #[case::binary(THRESHOLD, 0.5)]
    fn test_binary_recall(#[case] threshold: f64, #[case] expected: f64) {
        let input = dummy_classification_input(&ClassificationType::Binary).into();
        let mut metric = RecallMetric::binary(threshold);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value().unwrap().current()])
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
        TensorData::from([metric.value().unwrap().current()])
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
        TensorData::from([metric.value().unwrap().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default())
    }

    #[test]
    fn test_parameterized_unique_name() {
        let metric_a = RecallMetric::multiclass(1, ClassReduction::Macro);
        let metric_b = RecallMetric::multiclass(2, ClassReduction::Macro);
        let metric_c = RecallMetric::multiclass(1, ClassReduction::Macro);

        assert_ne!(metric_a.name(), metric_b.name());
        assert_eq!(metric_a.name(), metric_c.name());

        let metric_a = RecallMetric::binary(0.5);
        let metric_b = RecallMetric::binary(0.75);
        assert_ne!(metric_a.name(), metric_b.name());
    }

    #[test]
    fn test_recall_global_aggregation() {
        // Batch 1 (3 samples):
        //   preds:   [[0.9], [0.1], [0.1]] -> binary threshold 0.5 -> [1, 0, 0]
        //   targets: [[1],   [1],   [0]]   -> TP = 1, FN = 1 (denom = 2, Batch Recall = 50.0%)
        //
        // Batch 2 (6 samples):
        //   preds:   [[0.9], [0.1], [0.1], [0.1], [0.1], [0.1]] -> binary threshold 0.5 -> [1, 0, 0, 0, 0, 0]
        //   targets: [[1],   [1],   [1],   [1],   [1],   [0]]   -> TP = 1, FN = 4 (denom = 5, Batch Recall = 20.0%)
        //
        // Previously, using `NumericMetricState` would give a weighted average based on sample count N:
        //   (3 * 50.0% + 6 * 20.0%) / (3 + 6) = 270 / 9 = 30.0% (incorrectly weighting by total samples N)
        //
        // Correct Global Aggregation = Total TP / Total (TP + FN) = (1 + 1) / (2 + 5) = 2 / 7 = 28.5714%

        let mut metric = RecallMetric::binary(THRESHOLD);

        // Batch 1
        let input_batch1 = ConfusionStatsInput {
            predictions: Tensor::from([[0.9], [0.1], [0.1]]),
            targets: Tensor::from([[1], [1], [0]]),
        };
        let _ = metric.update(&input_batch1, &MetricMetadata::fake());

        // Batch 2
        let input_batch2 = ConfusionStatsInput {
            predictions: Tensor::from([[0.9], [0.1], [0.1], [0.1], [0.1], [0.1]]),
            targets: Tensor::from([[1], [1], [1], [1], [1], [0]]),
        };
        let _ = metric.update(&input_batch2, &MetricMetadata::fake());

        // Compute final aggregated metric
        let _final_entry = metric.compute();
        let global_recall = metric.final_value().current();

        let expected_global_recall = (2.0 / 7.0) * 100.0;

        TensorData::from([global_recall]).assert_approx_eq::<f32>(
            &TensorData::from([expected_global_recall]),
            Tolerance::default(),
        );
    }
}
