use super::classification::ClassReduction;
use burn_core::prelude::{Backend, Bool, Int, Tensor};
use std::fmt::{self, Debug};

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
    pub fn new(
        predictions: Tensor<B, 2>,
        targets: Tensor<B, 2, Bool>,
        threshold: Option<f64>,
        top_k: Option<usize>,
        class_reduction: ClassReduction,
    ) -> Self {
        let prediction_mask = match (threshold, top_k) {
            (Some(threshold), None) => {
                predictions.greater_elem(threshold)
            },
            (None, Some(top_k)) => {
                let mask = predictions.zeros_like();
                let values = predictions.ones_like().narrow(1, 0, top_k);
                let indexes = predictions.argsort_descending(1).narrow(1, 0, top_k);
                mask.scatter(1, indexes, values).bool()
            }
            _ => panic!("Either threshold (for binary or multilabel) or top_k (for multiclass) must be set."),
        };
        Self {
            confusion_classes: prediction_mask.int() + targets.int() * 2,
            class_reduction,
        }
    }

    /// sum over samples
    fn aggregate(
        sample_class_mask: Tensor<B, 2, Bool>,
        class_reduction: ClassReduction,
    ) -> Tensor<B, 1> {
        use ClassReduction::*;
        match class_reduction {
            Micro => sample_class_mask.float().sum(),
            Macro => sample_class_mask.float().sum_dim(0).squeeze(0),
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
    use super::{
        ClassReduction::{self, *},
        ConfusionStats,
    };
    use crate::tests::{
        dummy_classification_input,
        ClassificationType::{self, *},
        THRESHOLD,
    };
    use burn_core::prelude::TensorData;
    use rstest::rstest;

    #[rstest]
    #[should_panic]
    #[case::both_some(Some(THRESHOLD), Some(1))]
    #[should_panic]
    #[case::both_none(None, None)]
    fn test_exclusive_threshold_top_k(
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
    ) {
        let (predictions, targets) = dummy_classification_input(&Binary).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, Micro);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [1].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [3].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [1, 1, 1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [4].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [2, 1, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [5].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [2, 2, 1].into())]
    fn test_true_positive(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .true_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [2].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [8].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [2, 3, 3].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [4].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [1, 1, 2].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [3].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [0, 2, 1].into())]
    fn test_true_negative(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .true_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [1].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [2].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [1, 1, 0].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [6].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [2, 3, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [3].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [1, 1, 1].into())]
    fn test_false_positive(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .false_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [1].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [2].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [1, 0, 1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [1].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [0, 0, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [4].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [2, 0, 2].into())]
    fn test_false_negatives(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .false_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [2].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [5].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [2, 1, 2].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [5].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [2, 1, 2].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [9].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [4, 2, 3].into())]
    fn test_positive(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [3].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [3].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [10].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [3, 4, 3].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [10].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [3, 4, 3].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [6].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [1, 3, 2].into())]
    fn test_negative(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, Some(THRESHOLD), None, [2].into())]
    #[case::binary_macro(Binary, Macro, Some(THRESHOLD), None, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(1), [5].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(1), [2, 2, 1].into())]
    #[case::multiclass_micro(Multiclass, Micro, None, Some(2), [10].into())]
    #[case::multiclass_macro(Multiclass, Macro, None, Some(2), [4, 4, 2].into())]
    #[case::multilabel_micro(Multilabel, Micro, Some(THRESHOLD), None, [8].into())]
    #[case::multilabel_macro(Multilabel, Macro, Some(THRESHOLD), None, [3, 3, 2].into())]
    fn test_predicted_positive(
        #[case] classification_type: ClassificationType,
        #[case] class_reduction: ClassReduction,
        #[case] threshold: Option<f64>,
        #[case] top_k: Option<usize>,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&classification_type).into();
        ConfusionStats::new(predictions, targets, threshold, top_k, class_reduction)
            .predicted_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }
}
