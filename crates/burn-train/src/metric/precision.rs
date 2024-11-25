use super::{
    classification::ClassReduction,
    confusion_stats::ConfusionStats,
    state::{FormatOptions, NumericMetricState},
    Metric, MetricEntry, MetricMetadata, Numeric,
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::{cast::ToElement, Bool},
};
use core::marker::PhantomData;
use std::num::NonZeroUsize;

/// Input for precision metric.
#[derive(new, Debug, Clone)]
pub struct PrecisionInput<B: Backend> {
    /// Sample x Class Non thresholded normalized predictions.
    pub predictions: Tensor<B, 2>,
    /// Sample x Class one-hot encoded target.
    pub targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> From<PrecisionInput<B>> for (Tensor<B, 2>, Tensor<B, 2, Bool>) {
    fn from(input: PrecisionInput<B>) -> Self {
        (input.predictions, input.targets)
    }
}

impl<B: Backend> From<(Tensor<B, 2>, Tensor<B, 2, Bool>)> for PrecisionInput<B> {
    fn from(value: (Tensor<B, 2>, Tensor<B, 2, Bool>)) -> Self {
        Self::new(value.0, value.1)
    }
}

enum PrecisionConfig {
    Binary { threshold: f64 },
    Multiclass { top_k: NonZeroUsize },
    Multilabel { threshold: f64 },
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self::Binary { threshold: 0.5 }
    }
}

///The Precision Metric
#[derive(Default)]
pub struct PrecisionMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    class_reduction: ClassReduction,
    config: PrecisionConfig,
}

impl<B: Backend> PrecisionMetric<B> {
    /// Precision metric for binary classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn binary(threshold: f64) -> Self {
        Self {
            config: PrecisionConfig::Binary { threshold },
            ..Default::default()
        }
    }

    /// Precision metric for multiclass classification.
    ///
    /// # Arguments
    ///
    /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
    #[allow(dead_code)]
    pub fn multiclass(top_k: usize) -> Self {
        Self {
            config: PrecisionConfig::Multiclass {
                top_k: NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
            },
            ..Default::default()
        }
    }

    /// Precision metric for multi-label classification.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to transform a probability into a binary prediction.
    #[allow(dead_code)]
    pub fn multilabel(threshold: f64) -> Self {
        Self {
            config: PrecisionConfig::Multilabel { threshold },
            ..Default::default()
        }
    }

    /// Sets the class reduction method.
    #[allow(dead_code)]
    pub fn with_class_reduction(mut self, class_reduction: ClassReduction) -> Self {
        self.class_reduction = class_reduction;
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
    type Input = PrecisionInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let (predictions, targets) = input.clone().into();
        let [sample_size, _] = input.predictions.dims();

        let (threshold, top_k) = match self.config {
            PrecisionConfig::Binary { threshold } | PrecisionConfig::Multilabel { threshold } => {
                (Some(threshold), None)
            }
            PrecisionConfig::Multiclass { top_k } => (None, Some(top_k)),
        };

        let cf_stats =
            ConfusionStats::new(predictions, targets, threshold, top_k, self.class_reduction);
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
    use crate::tests::{dummy_classification_input, ClassificationType, THRESHOLD};
    use burn_core::tensor::TensorData;
    use rstest::rstest;

    #[rstest]
    #[case::binary_micro(Micro, THRESHOLD, 0.5)]
    #[case::binary_macro(Macro, THRESHOLD, 0.5)]
    fn test_binary_precision(
        #[case] class_reduction: ClassReduction,
        #[case] threshold: f64,
        #[case] expected: f64,
    ) {
        let input = dummy_classification_input(&ClassificationType::Binary).into();
        let mut metric = PrecisionMetric::binary(threshold).with_class_reduction(class_reduction);
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
        let input = dummy_classification_input(&ClassificationType::Multiclass).into();
        let mut metric = PrecisionMetric::multiclass(top_k).with_class_reduction(class_reduction);
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
        let input = dummy_classification_input(&ClassificationType::Multilabel).into();
        let mut metric =
            PrecisionMetric::multilabel(threshold).with_class_reduction(class_reduction);
        let _entry = metric.update(&input, &MetricMetadata::fake());
        TensorData::from([metric.value()])
            .assert_approx_eq(&TensorData::from([expected * 100.0]), 3)
    }
}
