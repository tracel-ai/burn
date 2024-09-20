use super::{AggregationType, ClassificationInput};
use burn_core::prelude::{Backend, Bool, Tensor};
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct ConfusionMatrix<B: Backend> {
    true_positive_mask: Tensor<B, 2, Bool>,
    false_positive_mask: Tensor<B, 2, Bool>,
    true_negative_mask: Tensor<B, 2, Bool>,
    false_negative_mask: Tensor<B, 2, Bool>,
    aggregation_type: AggregationType,
}

impl<B: Backend> Debug for ConfusionMatrix<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ratio_of_support_vec = |metric: Tensor<B, 1>| {
            self.clone()
                .ratio_of_support(metric)
                .to_data()
                .to_vec::<f32>()
                .unwrap()
        };
        f.debug_struct("ConfusionMatrix")
            .field("tp", &ratio_of_support_vec(self.clone().true_positive()))
            .field("fp", &ratio_of_support_vec(self.clone().false_positive()))
            .field("tn", &ratio_of_support_vec(self.clone().true_negative()))
            .field("fn", &ratio_of_support_vec(self.clone().false_negative()))
            .field(
                "support",
                &self.clone().support().to_data().to_vec::<f32>().unwrap(),
            )
            .finish()
    }
}

impl<B: Backend> ConfusionMatrix<B> {
    pub fn from(
        input: &ClassificationInput<B>,
        threshold: f64,
        aggregation_type: AggregationType,
    ) -> Self {
        let predictions = input.predictions.clone().greater_elem(threshold);
        let confusion_matrix = predictions.int() + input.targets.clone().int() * 2;
        Self {
            true_positive_mask: confusion_matrix.clone().equal_elem(3),
            false_negative_mask: confusion_matrix.clone().equal_elem(2),
            false_positive_mask: confusion_matrix.clone().equal_elem(1),
            true_negative_mask: confusion_matrix.equal_elem(0),
            aggregation_type,
        }
    }

    pub fn true_positive(self) -> Tensor<B, 1> {
        self.aggregation_type
            .aggregate(self.true_positive_mask.int())
    }

    pub fn true_negative(self) -> Tensor<B, 1> {
        self.aggregation_type
            .aggregate(self.true_negative_mask.int())
    }

    pub fn false_positive(self) -> Tensor<B, 1> {
        self.aggregation_type
            .aggregate(self.false_positive_mask.int())
    }

    pub fn false_negative(self) -> Tensor<B, 1> {
        self.aggregation_type
            .aggregate(self.false_negative_mask.int())
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
    use super::{AggregationType, ConfusionMatrix};
    use crate::metric::test::{dummy_classification_input, THRESHOLD, ClassificationType};
    use burn_core::tensor::TensorData;
    use strum::IntoEnumIterator;

    #[test]
    fn test_inner_representation() {
        for agg_type in AggregationType::iter() {
            for classification_type in ClassificationType::iter() {
                let (input, _) = dummy_classification_input(&classification_type);
                let conf_mat = ConfusionMatrix::from(&input, THRESHOLD, agg_type);
                let cm_mask = conf_mat.true_positive_mask.float()
                    + conf_mat.false_positive_mask.float()
                    + conf_mat.true_negative_mask.float()
                    + conf_mat.false_negative_mask.float();
                TensorData::assert_eq(&cm_mask.to_data(), &cm_mask.ones_like().to_data(), true);
            }
        }
    }

    #[test]
    fn test_confusion_matrix() {
        for agg_type in AggregationType::iter() {
            for classification_type in ClassificationType::iter() {
                let (input, target_diff) = dummy_classification_input(&classification_type);
                let conf_mat = ConfusionMatrix::from(&input, THRESHOLD, agg_type);
                TensorData::assert_eq(
                    &conf_mat.clone().false_negative().to_data(),
                    &agg_type
                        .aggregate(target_diff.clone().equal_elem(1.0).int())
                        .to_data(),
                    true,
                );
                TensorData::assert_eq(
                    &conf_mat.clone().false_positive().to_data(),
                    &agg_type
                        .aggregate(target_diff.clone().equal_elem(-1.0).int())
                        .to_data(),
                    true,
                );
                TensorData::assert_eq(
                    &conf_mat.clone().true_positive().to_data(),
                    &agg_type
                        .aggregate(target_diff.equal_elem(0).int() * input.targets.int())
                        .to_data(),
                    true,
                )
            }
        }
    }
}
