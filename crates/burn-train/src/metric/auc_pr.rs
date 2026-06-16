use super::MetricMetadata;
use super::state::{FormatOptions, PredictionAccumulatorState};
use crate::metric::{
    ClassReduction, ConfusionStatsInput, Metric, MetricAttributes, MetricName, Numeric,
    NumericAggregation, NumericAttributes, SerializedEntry,
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
    /// `AP = (1/P) · Σ_{positives, score desc} (cumulative positives / rank)`.
    ///
    /// `scores` and `targets` are `[n, c]` (`targets` as 0./1.); a column
    /// with no positive (`P = 0`) yields `NaN` (handled by the caller).
    fn average_precision(scores: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
        let [n, _c] = scores.dims();
        let device = scores.device();

        let order = scores.argsort_descending(0);
        let sorted_targets = targets.clone().gather(0, order);

        let tp = sorted_targets.clone().cumsum(0);

        let ranks = Tensor::<1, Int>::arange(1..n as i64 + 1, &device)
            .float()
            .reshape([n, 1]);

        let precision = tp / ranks;

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
        self.state
            .accumulate(input.predictions.clone(), input.targets.clone());

        // Recompute over the whole epoch: AP is rank-based, not a per-batch mean.
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

        self.state.update(
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
            aggregation: NumericAggregation::Last,
        }
        .into()
    }
}

impl Numeric for AucPrMetric {
    fn value(&self) -> super::NumericEntry {
        self.state.value()
    }

    fn running_value(&self) -> super::NumericEntry {
        self.state.value()
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
    /// (step-wise `AP = Σ (Rₙ−Rₙ₋₁)·Pₙ`). Scores are distinct so the
    /// estimator is unambiguous and matches sklearn exactly.
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
    #[case::multilabel_micro(Data::Multilabel, Micro, 0.5918017848017848)]
    fn test_auc_pr(
        #[case] data: Data,
        #[case] class_reduction: ClassReduction,
        #[case] expected: f64,
    ) {
        let mut metric = AucPrMetric::new(class_reduction);

        let _entry = metric.update(&input(data), &MetricMetadata::fake());

        TensorData::from([metric.value().current()])
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

        // Epoch value is independent of batching.
        TensorData::from([split.value().current()]).assert_approx_eq::<f64>(
            &TensorData::from([single.value().current()]),
            Tolerance::default(),
        );
    }
}
