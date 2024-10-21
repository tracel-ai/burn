use super::classification::ClassAverageType;
use burn_core::prelude::{Backend, Bool, Int, Tensor};
use burn_core::tensor::cast::ToElement;
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct ConfusionStats<B: Backend> {
    confusion_classes: Tensor<B, 2, Int>,
    class_average: ClassAverageType,
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
        f.debug_struct("ConfusionMatrix")
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
        threshold: f64,
        class_average: ClassAverageType,
    ) -> Self {
        let thresholded_predictions = predictions.greater_elem(threshold);
        Self {
            confusion_classes: thresholded_predictions.int() + targets.int() * 2,
            class_average,
        }
    }

    /// sum over samples
    fn aggregate(
        sample_class_mask: Tensor<B, 2, Bool>,
        class_average: ClassAverageType,
    ) -> Tensor<B, 1> {
        use ClassAverageType::*;
        match class_average {
            Micro => sample_class_mask.float().sum(),
            Macro => sample_class_mask.float().sum_dim(0).squeeze(0),
        }
    }

    ///convert to averaged metric, returns float
    fn average(mut aggregated_metric: Tensor<B, 1>, class_average: ClassAverageType) -> f64 {
        use ClassAverageType::*;
        let avg_tensor = match class_average {
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

    pub fn true_positive(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(3), self.class_average)
    }

    pub fn true_negative(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(0), self.class_average)
    }

    pub fn false_positive(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(1), self.class_average)
    }

    pub fn false_negative(self) -> Tensor<B, 1> {
        Self::aggregate(self.confusion_classes.equal_elem(2), self.class_average)
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

    pub fn precision(self) -> f64 {
        let class_average = self.class_average;
        Self::average(
            self.clone().true_positive() / self.predicted_positive(),
            class_average,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassAverageType::{self, *},
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
    #[case::binary_micro(Binary, Micro, [1].into())]
    #[case::binary_macro(Binary, Macro, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, [3].into())]
    #[case::multiclass_macro(Multiclass, Macro, [1, 1, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, [5].into())]
    #[case::multilabel_macro(Multilabel, Macro, [2, 2, 1].into())]
    fn test_true_positive(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .true_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [2].into())]
    #[case::binary_macro(Binary, Macro, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, [8].into())]
    #[case::multiclass_macro(Multiclass, Macro, [2, 3, 3].into())]
    #[case::multilabel_micro(Multilabel, Micro, [3].into())]
    #[case::multilabel_macro(Multilabel, Macro, [0, 2, 1].into())]
    fn test_true_negative(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .true_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [1].into())]
    #[case::binary_macro(Binary, Macro, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, [2].into())]
    #[case::multiclass_macro(Multiclass, Macro, [1, 1, 0].into())]
    #[case::multilabel_micro(Multilabel, Micro, [3].into())]
    #[case::multilabel_macro(Multilabel, Macro, [1, 1, 1].into())]
    fn test_false_positive(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .false_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [1].into())]
    #[case::binary_macro(Binary, Macro, [1].into())]
    #[case::multiclass_micro(Multiclass, Micro, [2].into())]
    #[case::multiclass_macro(Multiclass, Macro, [1, 0, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, [4].into())]
    #[case::multilabel_macro(Multilabel, Macro, [2, 0, 2].into())]
    fn test_false_negatives(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .false_negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [2].into())]
    #[case::binary_macro(Binary, Macro, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, [5].into())]
    #[case::multiclass_macro(Multiclass, Macro, [2, 1, 2].into())]
    #[case::multilabel_micro(Multilabel, Micro, [9].into())]
    #[case::multilabel_macro(Multilabel, Macro, [4, 2, 3].into())]
    fn test_positive(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [3].into())]
    #[case::binary_macro(Binary, Macro, [3].into())]
    #[case::multiclass_micro(Multiclass, Micro, [10].into())]
    #[case::multiclass_macro(Multiclass, Macro, [3, 4, 3].into())]
    #[case::multilabel_micro(Multilabel, Micro, [6].into())]
    #[case::multilabel_macro(Multilabel, Macro, [1, 3, 2].into())]
    fn test_negative(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .negative()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }

    #[rstest]
    #[case::binary_micro(Binary, Micro, [2].into())]
    #[case::binary_macro(Binary, Macro, [2].into())]
    #[case::multiclass_micro(Multiclass, Micro, [5].into())]
    #[case::multiclass_macro(Multiclass, Macro, [2, 2, 1].into())]
    #[case::multilabel_micro(Multilabel, Micro, [8].into())]
    #[case::multilabel_macro(Multilabel, Macro, [3, 3, 2].into())]
    fn test_predicted_positive(
        #[case] class_type: ClassificationType,
        #[case] avg_type: ClassAverageType,
        #[case] expected: Vec<i64>,
    ) {
        let (predictions, targets) = dummy_classification_input(&class_type).into();
        ConfusionStats::new(predictions, targets, THRESHOLD, avg_type)
            .predicted_positive()
            .int()
            .into_data()
            .assert_eq(&TensorData::from(expected.as_slice()), true);
    }
}
