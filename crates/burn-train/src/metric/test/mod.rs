use crate::metric::ClassificationInput;
use crate::TestBackend;
use burn_core::prelude::Tensor;
use burn_core::tensor::{Distribution, Shape};
use rand::seq::IteratorRandom;
use std::default::Default;
use strum::EnumIter;

/// Probability of tp before adding errors
pub const TRUE_POSITIVE_RATE: f64 = 0.5;
pub const THRESHOLD: f64 = 0.5;
pub const ERROR_PER_SAMPLE_RATE: f32 = 0.2;

#[derive(EnumIter, Debug)]
pub enum ClassificationType {
    Binary,
    Multiclass,
    Multilabel,
}

fn one_hot_encode(
    class_tensor: Tensor<TestBackend, 2>,
    n_classes: usize,
) -> Tensor<TestBackend, 2> {
    Tensor::stack(
        class_tensor
            .to_data()
            .iter()
            .map(|class_index: f32| {
                Tensor::<TestBackend, 1>::one_hot(
                    class_index as usize,
                    n_classes,
                    &class_tensor.device(),
                )
            })
            .collect(),
        0,
    )
}

/// Sample x Class shaped matrix for use in
/// classification metrics testing
pub fn dummy_classification_input(
    classification_type: &ClassificationType,
) -> (ClassificationInput<TestBackend>, Tensor<TestBackend, 2>) {
    let device = &Default::default();
    const N_SAMPLES: usize = 200;
    const N_CLASSES: usize = 4;

    let error_mask = {
        let mut rng = &mut rand::thread_rng();
        let change_idx = Tensor::from_floats(
            (0..N_SAMPLES)
                .into_iter()
                .choose_multiple(&mut rng, (N_SAMPLES as f32 * ERROR_PER_SAMPLE_RATE) as usize)
                .as_slice(),
            device,
        );
        let mut mask: Tensor<TestBackend, 1> = Tensor::zeros(Shape::new([N_SAMPLES]), device);
        let values = change_idx.ones_like();
        mask = mask.scatter(0, change_idx.int(), values);
        mask.unsqueeze_dim(1).bool()
    };

    let (targets, changed_targets) = match classification_type {
        ClassificationType::Binary => {
            let targets = Tensor::<TestBackend, 2>::random(
                Shape::new([N_SAMPLES, 1]),
                Distribution::Bernoulli(TRUE_POSITIVE_RATE),
                device,
            )
            .bool();

            let changed_targets = Tensor::equal(targets.clone(), error_mask).bool_not();
            (targets, changed_targets.float())
        }
        ClassificationType::Multiclass => {
            let mut classes_changes = (Tensor::<TestBackend, 2>::random(
                [N_SAMPLES, 2],
                Distribution::Default,
                device,
            ) * 4).int().float().chunk(2, 1);
            let (classes, changes) = (classes_changes.remove(0), classes_changes.remove(0));
            let changed_classes = (classes.clone() + changes.clamp(1, 2) * error_mask.clone().float()) % N_CLASSES as f32;
            (
                one_hot_encode(classes, N_CLASSES).bool(),
                one_hot_encode(changed_classes, N_CLASSES),
            )
        }
        ClassificationType::Multilabel => {
            let targets = Tensor::<TestBackend, 2>::random(
                [N_SAMPLES, N_CLASSES - 1],
                Distribution::Default,
                device,
            )
            .greater_elem(THRESHOLD);

            (targets.clone(), (targets.float() + error_mask.float()) % 2)
        }
    };
    let predictions = (changed_targets.random_like(Distribution::Uniform(0.0, THRESHOLD - 0.1))
        - changed_targets.clone())
    .abs();

    (
        ClassificationInput::new(predictions, targets.clone()),
        targets.float().sub(changed_targets),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::TensorData;
    use strum::IntoEnumIterator;
    use burn_core::tensor::cast::ToElement;

    #[test]
    fn test_predictions_targets_match() {
        for classification_type in ClassificationType::iter() {
            let (input, target_diff) = dummy_classification_input(&classification_type);
            let thresholded_prediction = input.predictions.clone().greater_elem(THRESHOLD);
            TensorData::assert_eq(
                &(target_diff.clone() + thresholded_prediction.float()).to_data(),
                &input.targets.clone().float().to_data(),
                true,
            );
        }
    }

    #[test]
    fn test_error_rate() {
        for classification_type in ClassificationType::iter() {
            let (_, target_diff) = dummy_classification_input(&classification_type);
            let mean_difference_targets = target_diff.abs().bool().any_dim(1).float().mean().into_scalar().to_f32();
            assert_eq!(
                mean_difference_targets,
                ERROR_PER_SAMPLE_RATE,
            );
        }
    }
}
