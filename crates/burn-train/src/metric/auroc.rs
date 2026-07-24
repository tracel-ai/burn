use core::f64;

use super::MetricMetadata;
use super::state::{FormatOptions, PredictionAccumulatorState};
use crate::metric::{
    ClassReduction, ConfusionStatsInput, Metric, MetricName, Numeric, SerializedEntry,
};
use burn_core::tensor::{Bool, Tensor};
use std::sync::Arc;

/// The Area Under the Receiver Operating Characteristic Curve (AUROC, also
/// referred to as [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)).
///
/// Supports binary, multiclass and multi-label classification through a
/// One-vs-Rest decomposition, aggregated with the configured
/// [class reduction](ClassReduction).
#[derive(Clone)]
pub struct AurocMetric {
    name: MetricName,
    state: PredictionAccumulatorState,
    class_reduction: ClassReduction,
}

impl Default for AurocMetric {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl AurocMetric {
    fn new(class_reduction: ClassReduction) -> Self {
        let state = Default::default();
        let name = Arc::new(format!("AUROC [{:?}]", class_reduction));

        Self {
            state,
            class_reduction,
            name,
        }
    }

    /// AUROC metric for binary classification.
    #[allow(dead_code)]
    pub fn binary() -> Self {
        Self::new(ClassReduction::default())
    }

    /// AUROC metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multiclass(class_reduction: ClassReduction) -> Self {
        Self::new(class_reduction)
    }

    /// AUROC metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multilabel(class_reduction: ClassReduction) -> Self {
        Self::new(class_reduction)
    }

    fn pairwise_auc(scores: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
        let [n, c] = scores.dims();

        let si = scores.clone().reshape([n, 1, c]);
        let sj = scores.reshape([1, n, c]);

        let yi = targets.clone().reshape([n, 1, c]);
        let yj = targets.reshape([1, n, c]);

        let valid: Tensor<3> = yi * (1.0 - yj);

        let reduce = |t: Tensor<3>| t.sum_dim(0).sum_dim(1).squeeze_dims::<1>(&[0, 1]);

        let num_pairs = reduce(valid.clone());
        let correct_pairs = reduce(si.clone().greater(sj.clone()).float() * valid.clone());
        let tied_pairs = reduce(si.equal(sj).float() * valid);

        (correct_pairs + 0.5 * tied_pairs) / num_pairs
    }

    fn compute_auc(&self, predictions: &Tensor<2>, targets: &Tensor<2, Bool>) -> f64 {
        let [n, c] = predictions.dims();

        let (scores, targets) = match self.class_reduction {
            ClassReduction::Macro => (predictions.clone(), targets.clone().float()),
            ClassReduction::Micro => (
                predictions.clone().reshape([n * c, 1]),
                targets.clone().float().reshape([n * c, 1]),
            ),
        };

        let auc = Self::pairwise_auc(scores, targets);

        let keep = auc
            .clone()
            .is_nan()
            .bool_not()
            .argwhere()
            .squeeze_dim::<1>(1);

        if keep.dims()[0] == 0 {
            log::warn!(
                "AUROC is undefined (no class has both positive and negative samples in the \
                 epoch); reporting 0.5 (chance level)."
            );
            return 0.5;
        }

        auc.select(0, keep).mean().into_scalar()
    }
}

impl Metric for AurocMetric {
    type Input = ConfusionStatsInput;

    fn update(
        &mut self,
        input: &ConfusionStatsInput,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
        // Update the state with predictions and targets
        self.state
            .accumulate(input.predictions.clone(), input.targets.clone());

        // Serialize placeholder to indicate no valid scalar exists yet mid-epoch
        self.state
            .serialize_placeholder(FormatOptions::new(self.name()).unit("%").precision(2))
    }

    fn compute(&mut self) -> SerializedEntry {
        // Guard against an empty epoch calculation
        if self.state.is_empty() {
            return self
                .state
                .serialize_placeholder(FormatOptions::new(self.name()).unit("%").precision(2));
        }

        // Recompute over the whole epoch
        let (predictions, targets) = self.state.tensors();
        let metric = self.compute_auc(&predictions, &targets);

        // Complete the state with the calculated scalar
        self.state.compute(
            100.0 * metric,
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

impl Numeric for AurocMetric {
    fn value(&self) -> Option<super::NumericEntry> {
        None // current value is invalid; requires epoch-level aggregation
    }

    fn running_value(&self) -> Option<super::NumericEntry> {
        None
    }

    fn final_value(&self) -> super::NumericEntry {
        self.state
            .value()
            .expect("Compute must be called to get final value")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::ClassReduction::{self, *};
    use burn_core::tensor::{TensorData, Tolerance};
    use rstest::rstest;

    /// Inputs and expected AUROC computed with an independent reference
    /// equivalent to scikit-learn's `roc_auc_score` (Mann-Whitney U:
    /// `(#pos>neg + 0.5·ties) / (P·N)`, One-vs-Rest, macro/micro). Scores
    /// are distinct so the statistic is unambiguous and matches sklearn.
    #[derive(Clone, Copy)]
    enum Data {
        Binary,
        Multiclass,
        Multilabel,
    }

    fn input(data: Data) -> ConfusionStatsInput {
        let dev = Default::default();
        match data {
            Data::Binary => ConfusionStatsInput::new(
                Tensor::from_data([[0.34], [0.64], [0.12], [0.19], [0.53], [0.38]], &dev),
                Tensor::from_data([[0], [0], [0], [0], [1], [1]], &dev),
            ),
            Data::Multiclass => ConfusionStatsInput::new(
                Tensor::from_data(
                    [
                        [0.79, 0.41, 0.16],
                        [0.25, 0.93, 0.78],
                        [0.61, 0.09, 0.21],
                        [0.9, 0.31, 0.33],
                        [0.16, 0.82, 0.57],
                        [0.57, 0.18, 0.63],
                    ],
                    &dev,
                ),
                Tensor::from_data(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                    ],
                    &dev,
                ),
            ),
            Data::Multilabel => ConfusionStatsInput::new(
                Tensor::from_data(
                    [
                        [0.11, 0.57, 0.9],
                        [0.13, 0.66, 0.37],
                        [0.71, 0.85, 0.6],
                        [0.29, 0.69, 0.49],
                        [0.68, 0.45, 0.25],
                        [0.33, 0.36, 0.31],
                    ],
                    &dev,
                ),
                Tensor::from_data(
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 1, 1],
                    ],
                    &dev,
                ),
            ),
        }
    }

    #[rstest]
    // Binary is a single column -> Macro == Micro.
    #[case::binary_macro(Data::Binary, Macro, 0.75)]
    #[case::binary_micro(Data::Binary, Micro, 0.75)]
    #[case::multiclass_macro(Data::Multiclass, Macro, 0.5666666666666667)]
    #[case::multiclass_micro(Data::Multiclass, Micro, 0.6458333333333333)]
    #[case::multilabel_macro(Data::Multilabel, Macro, 0.2907407407407407)]
    #[case::multilabel_micro(Data::Multilabel, Micro, 0.3611111111111111)]
    fn test_auroc(
        #[case] data: Data,
        #[case] class_reduction: ClassReduction,
        #[case] expected: f64,
    ) {
        let mut metric = AurocMetric::new(class_reduction);

        let _entry = metric.update(&input(data), &MetricMetadata::fake());
        let _entry = metric.compute();

        TensorData::from([metric.final_value().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default());
    }

    #[rstest]
    #[case::macro_reduction(Macro)]
    #[case::micro_reduction(Micro)]
    fn test_auroc_perfect_separation(#[case] class_reduction: ClassReduction) {
        let device = Default::default();
        let mut metric = AurocMetric::new(class_reduction);

        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device),
            Tensor::from_data([[0, 1], [1, 0], [1, 0], [0, 1]], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let _entry = metric.compute();
        assert_eq!(metric.final_value().current(), 100.0);
    }

    #[rstest]
    #[case::macro_reduction(Macro)]
    #[case::micro_reduction(Micro)]
    fn test_auroc_chance_level(#[case] class_reduction: ClassReduction) {
        let device = Default::default();
        let mut metric = AurocMetric::new(class_reduction);

        // All scores tied -> every pair is a tie -> AUROC = 0.5.
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], &device),
            Tensor::from_data([[0, 1], [1, 0], [1, 0], [0, 1]], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let _entry = metric.compute();
        assert_eq!(metric.final_value().current(), 50.0);
    }

    #[test]
    fn test_auroc_macro_drops_degenerate_class() {
        let device = Default::default();
        let mut metric = AurocMetric::new(Macro);

        // Class 2 never appears (column all-negative) -> its AUROC is undefined
        // and must be dropped, leaving the two well-separated classes at 1.0.
        let input = ConfusionStatsInput::new(
            Tensor::from_data(
                [
                    [0.9, 0.1, 0.0],
                    [0.2, 0.8, 0.0],
                    [0.7, 0.3, 0.0],
                    [0.1, 0.6, 0.0],
                ],
                &device,
            ),
            Tensor::from_data([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let _entry = metric.compute();
        assert_eq!(metric.final_value().current(), 100.0);
    }

    #[test]
    fn test_auroc_all_degenerate_is_chance() {
        let device = Default::default();
        let mut metric = AurocMetric::binary();

        // Only positives -> no valid pair in any column -> undefined ->
        // reported as chance level (0.5).
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.9], [0.8], [0.7], [0.6]], &device),
            Tensor::from_data([[1], [1], [1], [1]], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let _entry = metric.compute();
        assert_eq!(metric.final_value().current(), 50.0);
    }

    #[test]
    fn test_auroc_reduction_changes_name() {
        let macro_metric = AurocMetric::new(Macro);
        let micro_metric = AurocMetric::new(Micro);

        assert_ne!(macro_metric.name(), micro_metric.name());
    }

    #[test]
    fn test_auroc_accumulates_across_batches() {
        let dev = Default::default();

        // Whole dataset as a single batch.
        let mut single = AurocMetric::binary();
        single.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.9], [0.4], [0.8], [0.2], [0.6], [0.1]], &dev),
                Tensor::from_data([[1], [0], [1], [0], [1], [0]], &dev),
            ),
            &MetricMetadata::fake(),
        );
        single.compute();

        // Same dataset split across two batches.
        let mut split = AurocMetric::binary();
        split.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.9], [0.4], [0.8]], &dev),
                Tensor::from_data([[1], [0], [1]], &dev),
            ),
            &MetricMetadata::fake(),
        );
        split.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.2], [0.6], [0.1]], &dev),
                Tensor::from_data([[0], [1], [0]], &dev),
            ),
            &MetricMetadata::fake(),
        );
        split.compute();

        TensorData::from([split.final_value().current()]).assert_approx_eq::<f64>(
            &TensorData::from([single.final_value().current()]),
            Tolerance::default(),
        );
    }

    #[test]
    #[should_panic = "Compute must be called to get final value"]
    fn test_auroc_should_panic_before_compute() {
        let dev = Default::default();

        let mut split = AurocMetric::binary();
        split.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.9], [0.4], [0.8]], &dev),
                Tensor::from_data([[1], [0], [1]], &dev),
            ),
            &MetricMetadata::fake(),
        );

        // AUROC is not valid for a batch, and is not meaningful until all statistics have been accumulated
        assert!(split.value().is_none());
        assert!(split.running_value().is_none());

        split.final_value();
    }
}
