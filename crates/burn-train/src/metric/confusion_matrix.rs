use super::{ClassificationAverage, ClassificationInput};
use burn_core::prelude::{Backend, Int, Tensor};
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct ConfusionMatrix<B: Backend> {
    confusion_matrix_map: Tensor<B, 2, Int>,
    classification_average: ClassificationAverage,
}

impl<B: Backend> Debug for ConfusionMatrix<B> {
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

impl<B: Backend> ConfusionMatrix<B> {
    pub fn new(
        input: &ClassificationInput<B>,
        threshold: f64,
        classification_average: ClassificationAverage,
    ) -> Self {
        let thresholded_predictions = input.predictions.clone().greater_elem(threshold);
        Self {
            confusion_matrix_map: thresholded_predictions.int() + input.targets.clone().int() * 2,
            classification_average,
        }
    }

    pub fn true_positive(self) -> Tensor<B, 1> {
        self.classification_average
            .aggregate_sum(self.confusion_matrix_map.equal_elem(3))
    }

    pub fn true_negative(self) -> Tensor<B, 1> {
        self.classification_average
            .aggregate_sum(self.confusion_matrix_map.equal_elem(0))
    }

    pub fn false_positive(self) -> Tensor<B, 1> {
        self.classification_average
            .aggregate_sum(self.confusion_matrix_map.equal_elem(1))
    }

    pub fn false_negative(self) -> Tensor<B, 1> {
        self.classification_average
            .aggregate_sum(self.confusion_matrix_map.equal_elem(2))
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
        ClassificationAverage::{self, *},
        ConfusionMatrix,
    };
    use crate::{
        tests::{
            dummy_classification_input,
            ClassificationType::{self, *},
            THRESHOLD,
        },
        TestBackend, TestDevice,
    };
    use burn_core::prelude::{Tensor, TensorData};
    use yare::parameterized;

    #[parameterized(
    binary_micro = {Binary, Micro, [1].into()},
    binary_macro = {Binary, Macro, [1].into()},
    multiclass_micro = {Multiclass, Micro, [3].into()},
    multiclass_macro = {Multiclass, Macro, [1, 1, 1].into()},
    multilabel_micro = {Multilabel, Micro, [5].into()},
    multilabel_macro = {Multilabel, Macro, [2, 2, 1].into()})]
    fn test_true_positive(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .true_positive()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data()
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [2].into()},
    binary_macro = {Binary, Macro, [2].into()},
    multiclass_micro = {Multiclass, Micro, [8].into()},
    multiclass_macro = {Multiclass, Macro, [2, 3, 3].into()},
    multilabel_micro = {Multilabel, Micro, [3].into()},
    multilabel_macro = {Multilabel, Macro, [0, 2, 1].into()})]
    fn test_true_negative(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .true_negative()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data()
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [1].into()},
    binary_macro = {Binary, Macro, [1].into()},
    multiclass_micro = {Multiclass, Micro, [2].into()},
    multiclass_macro = {Multiclass, Macro, [1, 1, 0].into()},
    multilabel_micro = {Multilabel, Micro, [3].into()},
    multilabel_macro = {Multilabel, Macro, [1, 1, 1].into()})]
    fn test_false_positive(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .false_positive()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data(),
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [1].into()},
    binary_macro = {Binary, Macro, [1].into()},
    multiclass_micro = {Multiclass, Micro, [2].into()},
    multiclass_macro = {Multiclass, Macro, [1, 0, 1].into()},
    multilabel_micro = {Multilabel, Micro, [4].into()},
    multilabel_macro = {Multilabel, Macro, [2, 0, 2].into()})]
    fn test_false_negatives(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .false_negative()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data(),
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [2].into()},
    binary_macro = {Binary, Macro, [2].into()},
    multiclass_micro = {Multiclass, Micro, [5].into()},
    multiclass_macro = {Multiclass, Macro, [2, 1, 2].into()},
    multilabel_micro = {Multilabel, Micro, [9].into()},
    multilabel_macro = {Multilabel, Macro, [4, 2, 3].into()})]
    fn test_positive(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .positive()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data(),
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [3].into()},
    binary_macro = {Binary, Macro, [3].into()},
    multiclass_micro = {Multiclass, Micro, [10].into()},
    multiclass_macro = {Multiclass, Macro, [3, 4, 3].into()},
    multilabel_micro = {Multilabel, Micro, [6].into()},
    multilabel_macro = {Multilabel, Macro, [1, 3, 2].into()})]
    fn test_negative(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .negative()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data(),
        )
    }

    #[parameterized(
    binary_micro = {Binary, Micro, [2].into()},
    binary_macro = {Binary, Macro, [2].into()},
    multiclass_micro = {Multiclass, Micro, [5].into()},
    multiclass_macro = {Multiclass, Macro, [2, 2, 1].into()},
    multilabel_micro = {Multilabel, Micro, [8].into()},
    multilabel_macro = {Multilabel, Macro, [3, 3, 2].into()})]
    fn test_predicted_positive(
        class_type: ClassificationType,
        avg_type: ClassificationAverage,
        expected: TensorData,
    ) {
        let input = dummy_classification_input(&class_type);
        let test_value = ConfusionMatrix::new(&input, THRESHOLD, avg_type)
            .predicted_positive()
            .into_data();
        assert_eq!(
            test_value,
            Tensor::<TestBackend, 1>::from_data(expected, &TestDevice::default()).into_data(),
        )
    }
}
