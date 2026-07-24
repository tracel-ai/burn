use crate::metric::{MetricName, Numeric, state::ConfusionStatsState};

use super::{
    Metric, MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry, SerializedEntry,
    classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
    confusion_stats::{ConfusionStats, ConfusionStatsInput},
    state::FormatOptions,
};
use std::{num::NonZeroUsize, sync::Arc};

/// The [F-beta score](https://en.wikipedia.org/wiki/F-score) metric.
///
/// The `beta` parameter represents the ratio of recall importance to precision importance.
/// `beta > 1` gives more weight to recall, while `beta < 1` favors precision.
#[derive(Clone)]
pub struct FBetaScoreMetric {
    name: MetricName,
    state: ConfusionStatsState,
    config: ClassificationMetricConfig,
    beta: f64,
}

impl Default for FBetaScoreMetric {
    fn default() -> Self {
        Self::new(Default::default(), Default::default())
    }
}

impl FBetaScoreMetric {
    #[allow(dead_code)]
    fn new(config: ClassificationMetricConfig, beta: f64) -> Self {
        let name = Arc::new(format!(
            "FBetaScore ({}) @ {:?} [{:?}]",
            beta, config.decision_rule, config.class_reduction
        ));
        Self {
            name,
            config,
            beta,
            state: Default::default(),
        }
    }

    /// F-beta score metric for binary classification.
    ///
    /// # Arguments
    ///
    /// * `beta` - Positive real factor to weight recall's importance.
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn binary(beta: f64, threshold: f64) -> Self {
        Self::new(
            ClassificationMetricConfig {
                decision_rule: DecisionRule::Threshold(threshold),
                // binary classification results are the same independently of class_reduction
                ..Default::default()
            },
            beta,
        )
    }

    /// F-beta score metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `beta` - Positive real factor to weight recall's importance.
    /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multiclass(beta: f64, top_k: usize, class_reduction: ClassReduction) -> Self {
        Self::new(
            ClassificationMetricConfig {
                decision_rule: DecisionRule::TopK(
                    NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
                ),
                class_reduction,
            },
            beta,
        )
    }

    /// F-beta score metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `beta` - Positive real factor to weight recall's importance.
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multilabel(beta: f64, threshold: f64, class_reduction: ClassReduction) -> Self {
        Self::new(
            ClassificationMetricConfig {
                decision_rule: DecisionRule::Threshold(threshold),
                class_reduction,
            },
            beta,
        )
    }
}

impl Metric for FBetaScoreMetric {
    type Input = ConfusionStatsInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [sample_size, _] = input.predictions.dims();

        let stats = ConfusionStats::new(input, &self.config);
        let tp = Some(stats.clone().true_positive());
        let fp = Some(stats.clone().false_positive());
        let fn_ = Some(stats.false_negative());

        self.state.update(tp, fp, fn_, sample_size);
        self.state.compute_update(
            self.config.class_reduction,
            FormatOptions::new(self.name()).unit("%").precision(2),
            |tp, fp, fn_| {
                let (tp, fp, fn_) = (tp.unwrap(), fp.unwrap(), fn_.unwrap());
                let beta_sq = self.beta.powi(2);
                let scaled_tp = tp.clone() * (1.0 + beta_sq);
                let denom = scaled_tp.clone() + (fn_ * beta_sq) + fp;

                (scaled_tp / denom) * 100.0
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

impl Numeric for FBetaScoreMetric {
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
        FBetaScoreMetric, Metric, MetricMetadata,
    };
    use crate::metric::Numeric;
    use crate::tests::{ClassificationType, THRESHOLD, dummy_classification_input};
    use burn_core::tensor::TensorData;
    use burn_core::tensor::Tolerance;
    use rstest::rstest;

    #[rstest]
    #[case::binary_b1(1.0, THRESHOLD, 0.5)]
    #[case::binary_b2(2.0, THRESHOLD, 0.5)]
    fn test_binary_fscore(#[case] beta: f64, #[case] threshold: f64, #[case] expected: f64) {
        let input = dummy_classification_input(&ClassificationType::Binary).into();
        let mut metric = FBetaScoreMetric::binary(beta, threshold);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value().unwrap().current()])
            .assert_approx_eq::<f32>(&TensorData::from([expected * 100.0]), Tolerance::default())
    }

    #[rstest]
    #[case::multiclass_b1_micro_k1(1.0, Micro, 1, 3.0/5.0)]
    #[case::multiclass_b1_micro_k2(1.0, Micro, 2, 2.0/(5.0/4.0 + 10.0/4.0))]
    #[case::multiclass_b1_macro_k1(1.0, Macro, 1, (0.5 + 2.0/(1.0 + 2.0) + 2.0/(2.0 + 1.0))/3.0)]
    #[case::multiclass_b1_macro_k2(1.0, Macro, 2, (2.0/(1.0 + 2.0) + 2.0/(1.0 + 4.0) + 0.5)/3.0)]
    #[case::multiclass_b2_micro_k1(2.0, Micro, 1, 3.0/5.0)]
    #[case::multiclass_b2_micro_k2(2.0, Micro, 2, 5.0*4.0/(4.0*5.0 + 10.0))]
    #[case::multiclass_b2_macro_k1(2.0, Macro, 1, (0.5 + 5.0/(4.0 + 2.0) + 5.0/(8.0 + 1.0))/3.0)]
    #[case::multiclass_b2_macro_k2(2.0, Macro, 2, (5.0/(4.0 + 2.0) + 5.0/(4.0 + 4.0) + 0.5)/3.0)]
    fn test_multiclass_fscore(
        #[case] beta: f64,
        #[case] class_reduction: ClassReduction,
        #[case] top_k: usize,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multiclass).into();
        let mut metric = FBetaScoreMetric::multiclass(beta, top_k, class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value().unwrap().current()])
            .assert_approx_eq::<f32>(&TensorData::from([expected * 100.0]), Tolerance::default())
    }

    #[rstest]
    #[case::multilabel_micro(1.0, Micro, THRESHOLD, 2.0/(9.0/5.0 + 8.0/5.0))]
    #[case::multilabel_macro(1.0, Macro, THRESHOLD, (2.0/(2.0 + 3.0/2.0) + 2.0/(1.0 + 3.0/2.0) + 2.0/(3.0+2.0))/3.0)]
    #[case::multilabel_micro(2.0, Micro, THRESHOLD, 5.0/(4.0*9.0/5.0 + 8.0/5.0))]
    #[case::multilabel_macro(2.0, Macro, THRESHOLD, (5.0/(8.0 + 3.0/2.0) + 5.0/(4.0 + 3.0/2.0) + 5.0/(12.0+2.0))/3.0)]
    fn test_multilabel_fscore(
        #[case] beta: f64,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: f64,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Multilabel).into();
        let mut metric = FBetaScoreMetric::multilabel(beta, threshold, class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value().unwrap().current()])
            .assert_approx_eq::<f32>(&TensorData::from([expected * 100.0]), Tolerance::default())
    }

    #[test]
    fn test_parameterized_unique_name() {
        let metric_a = FBetaScoreMetric::multiclass(0.5, 1, ClassReduction::Macro);
        let metric_b = FBetaScoreMetric::multiclass(0.5, 2, ClassReduction::Macro);
        let metric_c = FBetaScoreMetric::multiclass(0.5, 1, ClassReduction::Macro);

        assert_ne!(metric_a.name(), metric_b.name());
        assert_eq!(metric_a.name(), metric_c.name());

        let metric_a = FBetaScoreMetric::binary(0.5, 0.5);
        let metric_b = FBetaScoreMetric::binary(0.75, 0.5);
        assert_ne!(metric_a.name(), metric_b.name());
    }
}
