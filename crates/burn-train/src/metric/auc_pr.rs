use super::MetricMetadata;
use super::state::{FormatOptions, PredictionAccumulatorState};
use crate::metric::{
    ClassReduction, ConfusionStatsInput, Metric, MetricAttributes, MetricName, Numeric,
    NumericAttributes, SerializedEntry,
};
use burn_core::tensor::{Int, Tensor};
use std::sync::Arc;

/// The Area Under the Precision-Recall Curve (AUC-PR).
///
/// Computed as **Average Precision** — `AP = Σ (Rₙ − Rₙ₋₁) · Pₙ` — the
/// standard non-interpolated estimator of the area under the
/// precision-recall curve (equivalent to scikit-learn's
/// `average_precision_score`), not the (biased) trapezoidal integration.
///
/// Supports binary, multiclass and multi-label classification through a
/// One-vs-Rest decomposition, aggregated with the configured
/// [class reduction](ClassReduction).
#[derive(Clone)]
pub struct AucPrMetric {
    name: MetricName,
    state: PredictionAccumulatorState,
    class_reduction: ClassReduction,
}

impl Default for AucPrMetric {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl AucPrMetric {
    fn new(class_reduction: ClassReduction) -> Self {
        let state = Default::default();
        let name = Arc::new(format!("AUC-PR [{:?}]", class_reduction));

        Self {
            state,
            class_reduction,
            name,
        }
    }

    /// AUC-PR metric for binary classification.
    #[allow(dead_code)]
    pub fn binary() -> Self {
        Self::new(ClassReduction::default())
    }

    /// AUC-PR metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multiclass(class_reduction: ClassReduction) -> Self {
        Self::new(class_reduction)
    }

    /// AUC-PR metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `class_reduction` - [Class reduction](ClassReduction) type.
    #[allow(dead_code)]
    pub fn multilabel(class_reduction: ClassReduction) -> Self {
        Self::new(class_reduction)
    }

    /// Per-column Average Precision via the step-wise estimator
    /// `AP = (1/P) · Σ_{positives} precision at the positive sample's score threshold`.
    ///
    /// `scores` and `targets` are `[n, c]` (`targets` as 0./1.); a column
    /// with no positive (`P = 0`) yields `NaN` (handled by the caller).
    fn average_precision(scores: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
        let [n, c] = scores.dims();
        let device = scores.device();

        let (sorted_scores, order) = scores.sort_descending_with_indices(0);
        let sorted_targets = targets.clone().gather(0, order);

        let tp = sorted_targets.clone().cumsum(0);

        let ranks = Tensor::<1, Int>::arange(1..n as i64 + 1, &device).reshape([n, 1]);
        let mut precision = tp / ranks.clone().float();

        if n > 1 {
            // A threshold includes all samples with the same score. Every positive
            // in a tied group therefore contributes the precision at that group's end.
            let same_as_next = sorted_scores
                .clone()
                .narrow(0, 0, n - 1)
                .equal(sorted_scores.narrow(0, 1, n - 1));
            let non_end = Tensor::cat(vec![same_as_next, Tensor::zeros([1, c], &device)], 0);
            let group_end = (ranks - 1)
                .expand([n, c])
                .mask_fill(non_end, n as i64 - 1)
                // Propagate each end index backwards through its group. The final
                // row is always an end, so every gathered index is valid.
                .flip([0])
                .cummin(0)
                .flip([0]);
            precision = precision.gather(0, group_end);
        }

        let p_total = targets.sum_dim(0);
        let delta_recall = sorted_targets / p_total;

        (precision * delta_recall)
            .sum_dim(0)
            .squeeze_dims::<1>(&[0])
    }
}

impl Metric for AucPrMetric {
    type Input = ConfusionStatsInput;

    fn update(
        &mut self,
        input: &ConfusionStatsInput,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
        // Update the state
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

        // Recompute over the whole epoch: AP is rank-based.
        let (predictions, targets) = self.state.tensors();
        let [n, c] = predictions.dims();

        let (scores, targets) = match self.class_reduction {
            ClassReduction::Macro => (predictions, targets.float()),
            ClassReduction::Micro => (
                predictions.reshape([n * c, 1]),
                targets.float().reshape([n * c, 1]),
            ),
        };

        let ap = Self::average_precision(scores, targets);

        let keep = ap
            .clone()
            .is_nan()
            .bool_not()
            .argwhere()
            .squeeze_dim::<1>(1);

        let metric = if keep.dims()[0] == 0 {
            log::warn!(
                "AUC-PR is undefined (no class has positive samples in the epoch); reporting \
                 0.5 as a neutral fallback."
            );
            0.5
        } else {
            ap.select(0, keep).mean().into_scalar()
        };

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

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for AucPrMetric {
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

    /// Inputs and expected Average Precision computed with an independent
    /// reference equivalent to scikit-learn's `average_precision_score`
    /// (step-wise `AP = Σ (Rₙ−Rₙ₋₁)·Pₙ`). The multilabel micro case includes
    /// ties across columns, which must be treated as a single threshold.
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
                Tensor::from_data([[0.63], [0.25], [0.71], [0.3], [0.07], [0.66]], &dev),
                Tensor::from_data([[0], [1], [0], [0], [0], [0]], &dev),
            ),
            Data::Multiclass => ConfusionStatsInput::new(
                Tensor::from_data(
                    [
                        [0.45, 0.3, 0.36],
                        [0.83, 0.24, 0.09],
                        [0.19, 0.39, 0.29],
                        [0.3, 0.14, 0.46],
                        [0.73, 0.74, 0.16],
                        [0.43, 0.37, 0.88],
                    ],
                    &dev,
                ),
                Tensor::from_data(
                    [
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
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
                        [0.1, 0.73, 0.84],
                        [0.84, 0.74, 0.24],
                        [0.13, 0.54, 0.54],
                        [0.49, 0.48, 0.71],
                        [0.9, 0.17, 0.43],
                        [0.11, 0.29, 0.23],
                    ],
                    &dev,
                ),
                Tensor::from_data(
                    [
                        [1, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                    ],
                    &dev,
                ),
            ),
        }
    }

    #[rstest]
    // Binary is a single column -> Macro == Micro.
    #[case::binary_macro(Data::Binary, Macro, 0.2)]
    #[case::binary_micro(Data::Binary, Micro, 0.2)]
    #[case::multiclass_macro(Data::Multiclass, Macro, 0.6319444444444444)]
    #[case::multiclass_micro(Data::Multiclass, Micro, 0.379975579975580)]
    #[case::multilabel_macro(Data::Multilabel, Macro, 0.5944444444444444)]
    #[case::multilabel_micro(Data::Multilabel, Micro, 0.550135118149824)]
    fn test_auc_pr(
        #[case] data: Data,
        #[case] class_reduction: ClassReduction,
        #[case] expected: f64,
    ) {
        let mut metric = AucPrMetric::new(class_reduction);

        let _entry = metric.update(&input(data), &MetricMetadata::fake());
        let _entry = metric.compute();

        TensorData::from([metric.final_value().current()])
            .assert_approx_eq::<f64>(&TensorData::from([expected * 100.0]), Tolerance::default());
    }

    #[test]
    fn test_auc_pr_accumulates_across_batches() {
        let dev = Default::default();

        // Whole dataset as a single batch.
        let mut single = AucPrMetric::binary();
        single.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.9], [0.4], [0.8], [0.2], [0.6], [0.1]], &dev),
                Tensor::from_data([[1], [0], [1], [0], [1], [0]], &dev),
            ),
            &MetricMetadata::fake(),
        );
        single.compute();

        // Same dataset split across two batches.
        let mut split = AucPrMetric::binary();
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

    #[rstest]
    #[case([[1], [0]])]
    #[case([[0], [1]])]
    fn test_auc_pr_ties_are_order_and_batch_invariant(#[case] labels: [[i32; 1]; 2]) {
        let dev = Default::default();
        let input = ConfusionStatsInput::new(
            Tensor::from_data([[0.5], [0.5]], &dev),
            Tensor::from_data(labels, &dev),
        );

        for batch_size in [1, 2] {
            let mut metric = AucPrMetric::binary();
            for start in (0..2).step_by(batch_size) {
                metric.update(
                    &ConfusionStatsInput::new(
                        input.predictions.clone().narrow(0, start, batch_size),
                        input.targets.clone().narrow(0, start, batch_size),
                    ),
                    &MetricMetadata::fake(),
                );
            }
            metric.compute();

            // Both samples enter at the same threshold: precision = 1/2, recall = 1.
            assert_eq!(metric.final_value().current(), 50.0);
        }
    }

    #[rstest]
    #[case(Macro, 0.7)]
    #[case(Micro, 0.6211640211640211)]
    fn test_auc_pr_ties_with_class_reduction(
        #[case] reduction: ClassReduction,
        #[case] expected: f64,
    ) {
        let dev = Default::default();
        // Different tie boundaries in each column; micro reduction also ties
        // scores across columns. Expected values checked with scikit-learn.
        let input = ConfusionStatsInput::new(
            Tensor::from_data(
                [[0.9, 0.5], [0.9, 0.8], [0.5, 0.8], [0.5, 0.2], [0.1, 0.2]],
                &dev,
            ),
            Tensor::from_data([[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]], &dev),
        );
        let mut metric = AucPrMetric::multilabel(reduction);
        metric.update(&input, &MetricMetadata::fake());
        metric.compute();
        TensorData::from([metric.final_value().current()]).assert_approx_eq::<f64>(
            &TensorData::from([expected * 100.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_average_precision_matches_threshold_reference() {
        let dev = Default::default();
        let mut scores = vec![Vec::new(); 4];
        let mut targets = vec![Vec::new(); 4];
        let mut expected = Vec::new();

        // Exhaust all four-sample columns with scores in {0, 0.5, 1} and at least
        // one positive label. This includes every permutation of every tie pattern.
        for score_pattern in 0..81 {
            let column: [f32; 4] =
                core::array::from_fn(|i| ((score_pattern / 3usize.pow(i as u32)) % 3) as f32 / 2.0);
            for label_pattern in 1..16usize {
                let labels: [bool; 4] = core::array::from_fn(|i| label_pattern & (1 << i) != 0);

                // Independent O(n^2) reference: for each positive, measure precision
                // among ALL samples at or above its score, then average over positives.
                // No sorting or cumulative sums are used in this reference.
                let mut ap = 0.0;
                for i in 0..4 {
                    if labels[i] {
                        let predicted = (0..4).filter(|&j| column[j] >= column[i]).count();
                        let correct = (0..4)
                            .filter(|&j| labels[j] && column[j] >= column[i])
                            .count();
                        ap += correct as f32 / predicted as f32;
                    }
                    scores[i].push(column[i]);
                    targets[i].push(if labels[i] { 1.0f32 } else { 0.0 });
                }
                expected.push(ap / label_pattern.count_ones() as f32);
            }
        }

        let num_columns = expected.len();
        let scores = Tensor::from_data(
            TensorData::new(
                scores.into_iter().flatten().collect::<Vec<_>>(),
                [4, num_columns],
            ),
            &dev,
        );
        let targets = Tensor::from_data(
            TensorData::new(
                targets.into_iter().flatten().collect::<Vec<_>>(),
                [4, num_columns],
            ),
            &dev,
        );
        AucPrMetric::average_precision(scores, targets)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(expected, [num_columns]),
                Tolerance::absolute(1e-6),
            );
    }

    #[test]
    fn test_average_precision_single_sample_and_no_positives() {
        let dev = Default::default();
        let ap = AucPrMetric::average_precision(
            Tensor::from_data([[0.5, 0.5]], &dev),
            Tensor::from_data([[1.0, 0.0]], &dev),
        );
        assert_eq!(ap.clone().narrow(0, 0, 1).into_scalar::<f32>(), 1.0);
        assert!(ap.narrow(0, 1, 1).into_scalar::<f32>().is_nan());
    }

    #[test]
    #[should_panic = "Compute must be called to get final value"]
    fn test_auc_pr_should_panic_before_compute() {
        let dev = Default::default();

        let mut split = AucPrMetric::binary();
        split.update(
            &ConfusionStatsInput::new(
                Tensor::from_data([[0.9], [0.4], [0.8]], &dev),
                Tensor::from_data([[1], [0], [1]], &dev),
            ),
            &MetricMetadata::fake(),
        );

        // AUC-PR is not valid for a batch, and is not meaningful until all statistics have been accumulated
        assert!(split.value().is_none());
        assert!(split.running_value().is_none());

        split.final_value();
    }
}
