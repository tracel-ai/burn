use super::AggregationType;
use burn_core::prelude::{Backend, Bool, Tensor};
use burn_core::tensor::cast::ToElement;
use burn_core::tensor::Int;
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct ConfusionMatrix<B: Backend> {
    true_positive_mask: Tensor<B, 2, Bool>,
    false_positive_mask: Tensor<B, 2, Bool>,
    true_negative_mask: Tensor<B, 2, Bool>,
    false_negative_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> Debug for ConfusionMatrix<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let agg_type = AggregationType::Micro;
        f.debug_struct("ConfusionMatrix")
            .field(
                "tp",
                &(&self.clone().true_positive(agg_type).into_scalar().to_f64()
                    / &self.clone().target_support(agg_type)),
            )
            .field(
                "fp",
                &(&self.clone().false_positive(agg_type).into_scalar().to_f64()
                    / &self.clone().target_support(agg_type)),
            )
            .field(
                "tn",
                &(&self.clone().true_negative(agg_type).into_scalar().to_f64()
                    / &self.clone().target_support(agg_type)),
            )
            .field(
                "fn",
                &(&self.clone().false_negative(agg_type).into_scalar().to_f64()
                    / &self.clone().target_support(agg_type)),
            )
            .finish()
    }
}

/// Sample x Class tensors of thresholded model predictions
/// and targets
#[derive(new)]
pub struct ConfusionMatrixInput<B: Backend> {
    predictions: Tensor<B, 2, Bool>,
    targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> ConfusionMatrix<B> {
    pub fn from(input: ConfusionMatrixInput<B>) -> Self {
        let confusion_matrix = input.predictions.int() + input.targets.int() * 2;
        Self {
            true_positive_mask: confusion_matrix.clone().equal_elem(3),
            false_negative_mask: confusion_matrix.clone().equal_elem(2),
            false_positive_mask: confusion_matrix.clone().equal_elem(1),
            true_negative_mask: confusion_matrix.equal_elem(0),
        }
    }

    pub fn true_positive(self, aggregation_type: AggregationType) -> Tensor<B, 1, Int> {
        aggregation_type.aggregate(self.true_positive_mask.int())
    }

    pub fn true_negative(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        average_type.aggregate(self.true_negative_mask.int())
    }

    pub fn false_positive(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        average_type.aggregate(self.false_positive_mask.int())
    }

    pub fn false_negative(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        average_type.aggregate(self.false_negative_mask.int())
    }

    pub fn positive(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        self.clone().true_positive(average_type) + self.false_negative(average_type)
    }

    pub fn negative(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        self.clone().true_negative(average_type) + self.false_positive(average_type)
    }

    pub fn target_support(self, aggregation_type: AggregationType) -> f64 {
        aggregation_type.to_averaged_metric(
            (self.clone().positive(aggregation_type) + self.negative(aggregation_type)).float(),
        )
    }

    pub fn predicted_positive(self, average_type: AggregationType) -> Tensor<B, 1, Int> {
        self.clone().true_positive(average_type) + self.false_positive(average_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::test::test_prediction;
    use crate::metric::ClassificationType;

    #[test]
    fn test_true_positive() {
        let test_prediction = test_prediction(ClassificationType::Multilabel);
        let input = ConfusionMatrixInput::new(test_prediction.clone(), test_prediction);
        let conf_mat = ConfusionMatrix::from(input);
        print!("{:?}", conf_mat)
    }
}
