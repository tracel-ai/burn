use super::classification::{ClassReduction, ClassificationMetricConfig, DecisionRule};
use burn_core::prelude::{Backend, Bool, Int, Tensor};
use std::fmt::{self, Debug};

/// Input for confusion statistics error types.
#[derive(new, Debug, Clone)]
pub struct ConfusionStatsInput<B: Backend> {
    /// Sample x Class Non thresholded normalized predictions.
    pub predictions: Tensor<B, 2>,
    /// Sample x Class one-hot encoded target.
    pub targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> From<ConfusionStatsInput<B>> for (Tensor<B, 2>, Tensor<B, 2, Bool>) {
    fn from(input: ConfusionStatsInput<B>) -> Self {
        (input.predictions, input.targets)
    }
}

impl<B: Backend> From<(Tensor<B, 2>, Tensor<B, 2, Bool>)> for ConfusionStatsInput<B> {
    fn from(value: (Tensor<B, 2>, Tensor<B, 2, Bool>)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[derive(Clone)]
pub struct ConfusionStats<B: Backend> {
    confusion_classes: Tensor<B, 2, Int>,
    class_reduction: ClassReduction,
}

impl<B: Backend> Debug for ConfusionStats<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let to_vec = |tensor_data: Tensor<B, 1>| {
            tensor_data
                .to_data()
                .to_vec::<f32>()
                .expect("A vector representation of the input Tensor is expected")
        };
        let ratio_of_support_vec =
            |metric: Tensor<B, 1>| to_vec(self.clone().ratio_of_support(metric));
        f.debug_struct("ConfusionStats")
            .field("tp", &ratio_of_support_vec(self.clone().true_positive()))
            .field("fp", &ratio_of_support_vec(self.clone().false_positive()))
            .field("tn", &ratio_of_support_vec(self.clone().true_negative()))
            .field("fn", &ratio_of_support_vec(self.clone().false_negative()))
            .field("support", &to_vec(self.clone().support()))
            .finish()
    }
}

impl<B: Backend> ConfusionStats<B> {
    /// Expects `predictions` to be normalized.
    pub fn new(input: &ConfusionStatsInput<B>, config: &ClassificationMetricConfig) -> Self {
        let prediction_mask = match config.decision_rule {
            DecisionRule::Threshold(threshold) => input.predictions.clone().greater_elem(threshold),
            DecisionRule::TopK(top_k) => {
                let mask = input.predictions.zeros_like();
                let indexes =
                    input
                        .predictions
                        .clone()
                        .argsort_descending(1)
                        .narrow(1, 0, top_k.get());
                let values = indexes.ones_like().float();
                mask.scatter_add(1, indexes, values).bool()
            }
        };
        Self {
            confusion_classes: prediction_mask.int() + input.targets.clone().int() * 2,
            class_reduction: config.class_reduction,
        }
    }

    /// sum over samples
    fn aggregate(
        sample_class_mask: Tensor<B, 2, Bool>,
        class_reduction: ClassReduction,
    ) -> Tensor<B, 1> {
        use ClassReduction::{Macro, Micro};
        match class_reduction {
            Micro => sample_class_mask.float().sum(),
            Macro => sample_class_mask.float().sum_dim(0).squeeze_dim(0),
        }
    }

    pub fn true_positive(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(3), self.class_reduction)
    }

    pub fn true_negative(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(0), self.class_reduction)
    }

    pub fn false_positive(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(1), self.class_reduction)
    }

    pub fn false_negative(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(2), self.class_reduction)
    }

    pub fn positive(self) -> Tensor<B, 1> {
        self.clone().true_positive() + self.false_negative()
    }

    pub fn negative(self) -> Tensor<B, 1> {
        self.clone().true_negative() + self.false_positive()
    }

    pub fn predicted_positive(self) -> Tensor<B, 1> {
        self.clone().true_positive() + self.false_positive()
    }

    pub fn support(self) -> Tensor<B, 1> {
        self.clone().positive() + self.negative()
    }

    pub fn ratio_of_support(self, metric: Tensor<B, 1>) -> Tensor<B, 1> {
        metric / self.clone().support()
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfusionStats, ConfusionStatsInput};
    use crate::{
        TestBackend,
        metric::classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
        tests::{ClassificationType, THRESHOLD, dummy_classification_input},
    };
    use burn_core::prelude::TensorData;
    use rstest::{fixture, rstest};
    use std::num::NonZeroUsize;

    fn top_k_config(
        top_k: NonZeroUsize,
        class_reduction: ClassReduction,
    ) -> ClassificationMetricConfig {
        ClassificationMetricConfig {
            decision_rule: DecisionRule::TopK(top_k),
            class_reduction,
        }
    }
    #[fixture]
    #[once]
    fn top_k_config_k1_micro() -> ClassificationMetricConfig {
        top_k_config(NonZeroUsize::new(1).unwrap(), ClassReduction::Micro)
    }

    #[fixture]
    #[once]
    fn top_k_config_k1_macro() -> ClassificationMetricConfig {
        top_k_config(NonZeroUsize::new(1).unwrap(), ClassReduction::Macro)
    }
    #[fixture]
    #[once]
    fn top_k_config_k2_micro() -> ClassificationMetricConfig {
        top_k_config(NonZeroUsize::new(2).unwrap(), ClassReduction::Micro)
    }
    #[fixture]
    #[once]
    fn top_k_config_k2_macro() -> ClassificationMetricConfig {
        top_k_config(NonZeroUsize::new(2).unwrap(), ClassReduction::Macro)
    }

    fn threshold_config(
        threshold: f64,
        class_reduction: ClassReduction,
    ) -> ClassificationMetricConfig {
        ClassificationMetricConfig {
            decision_rule: DecisionRule::Threshold(threshold),
            class_reduction,
        }
    }
    #[fixture]
    #[once]
    fn threshold_config_micro() -> ClassificationMetricConfig {
        threshold_config(THRESHOLD, ClassReduction::Micro)
    }
    #[fixture]
    #[once]
    fn threshold_config_macro() -> ClassificationMetricConfig {
        threshold_config(THRESHOLD, ClassReduction::Macro)
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [1].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [3].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [1, 1, 1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [4].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [2, 1, 1].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [5].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [2, 2, 1].into())]
    fn test_true_positive(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .true_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [2].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [2].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [8].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [2, 3, 3].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [4].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [1, 1, 2].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [3].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [0, 2, 1].into())]
    fn test_true_negative(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .true_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [1].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [2].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [1, 1, 0].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [6].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [2, 3, 1].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [3].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [1, 1, 1].into())]
    fn test_false_positive(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .false_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [1].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [2].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [1, 0, 1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [1].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [0, 0, 1].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [4].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [2, 0, 2].into())]
    fn test_false_negatives(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .false_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [2].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [2].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [5].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [2, 1, 2].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [5].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [2, 1, 2].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [9].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [4, 2, 3].into())]
    fn test_positive(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [3].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [3].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [10].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [3, 4, 3].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [10].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [3, 4, 3].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [6].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [1, 3, 2].into())]
    fn test_negative(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(ClassificationType::Binary, threshold_config_micro(), [2].into())]
    #[case::binary_macro(ClassificationType::Binary, threshold_config_macro(), [2].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k1_micro(), [5].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k1_macro(), [2, 2, 1].into())]
    #[case::multiclass_micro(ClassificationType::Multiclass, top_k_config_k2_micro(), [10].into())]
    #[case::multiclass_macro(ClassificationType::Multiclass, top_k_config_k2_macro(), [4, 4, 2].into())]
    #[case::multilabel_micro(ClassificationType::Multilabel, threshold_config_micro(), [8].into())]
    #[case::multilabel_macro(ClassificationType::Multilabel, threshold_config_macro(), [3, 3, 2].into())]
    fn test_predicted_positive(
        #[case] classification_type: ClassificationType,
        #[case] config: ClassificationMetricConfig,
        #[case] expected: Vec<i64>,
    ) {
        let input: ConfusionStatsInput<TestBackend> =
            dummy_classification_input(&classification_type).into();
        ConfusionStats::new(&input, &config)
            .predicted_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }
}
