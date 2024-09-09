use crate::metric::ClassificationType;
use crate::TestBackend;
use burn_core::prelude::Tensor;
use burn_core::tensor::{Distribution, Shape, Bool};

/// Sample x Class shaped matrix for use in
/// classification metrics testing
pub fn test_prediction(classification_type: ClassificationType) -> Tensor<TestBackend, 2, Bool> {
    let device = &Default::default();
    const N_SAMPLES: usize = 100;
    const N_CLASSES: usize = 4;

    match classification_type {
        ClassificationType::Binary => Tensor::<TestBackend, 2>::random(Shape::new([N_SAMPLES, 1]), Distribution::Bernoulli(0.6), device).bool(),
        ClassificationType::Multiclass => {
            let classes: Tensor<TestBackend, 2> = Tensor::random(Shape::new([N_SAMPLES, N_CLASSES - 1]), Distribution::Bernoulli(0.6), device).sum_dim(1);
            Tensor::stack(classes.to_data().iter().map(|class_index: f64| {Tensor::<TestBackend, 1>::one_hot(class_index as usize, N_CLASSES, device)}).collect(), 0).bool()
        }
        ClassificationType::Multilabel => Tensor::<TestBackend, 2>::random(Shape::new([N_SAMPLES, N_CLASSES]), Distribution::Bernoulli(0.6), device).bool()
    }
}
