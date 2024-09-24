#![warn(missing_docs)]

//! A library for training neural networks using the burn crate.

#[macro_use]
extern crate derive_new;

/// The checkpoint module.
pub mod checkpoint;

pub(crate) mod components;

/// Renderer modules to display metrics and training information.
pub mod renderer;

/// The logger module.
pub mod logger;

/// The metric module.
pub mod metric;

mod learner;

pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(test)]
pub(crate) type TestDevice = burn_ndarray::NdArrayDevice;

#[cfg(test)]
pub(crate) mod tests {
    use crate::{metric::ClassificationInput, TestBackend, TestDevice};
    use burn_core::{
        prelude::{Bool, Tensor},
        tensor::Distribution,
    };
    use std::default::Default;

    /// Probability of tp before adding errors
    pub const THRESHOLD: f64 = 0.5;

    #[derive(Debug)]
    pub enum ClassificationType {
        Binary,
        Multiclass,
        Multilabel,
    }

    /// Sample x Class shaped matrix for use in
    /// classification metrics testing
    pub fn dummy_classification_input(
        classification_type: &ClassificationType,
    ) -> ClassificationInput<TestBackend> {
        let (real_targets, prediction_targets) = match classification_type {
            ClassificationType::Binary => {
                let real_targets = Tensor::<TestBackend, 2, Bool>::from_data(
                    [[0], [1], [0], [0], [1]],
                    &TestDevice::default(),
                );

                let prediction_targets = Tensor::<TestBackend, 2>::from_data(
                    [[0], [0], [1], [0], [1]],
                    &TestDevice::default(),
                );
                (real_targets, prediction_targets)
            }
            ClassificationType::Multiclass => {
                let real_targets = Tensor::<TestBackend, 2, Bool>::from_data(
                    [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
                    &TestDevice::default(),
                );

                let prediction_targets = Tensor::<TestBackend, 2>::from_data(
                    [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]],
                    &TestDevice::default(),
                );
                (real_targets, prediction_targets)
            }
            ClassificationType::Multilabel => {
                let real_targets = Tensor::<TestBackend, 2, Bool>::from_data(
                    [[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]],
                    &TestDevice::default(),
                );

                let prediction_targets = Tensor::<TestBackend, 2>::from_data(
                    [[0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]],
                    &TestDevice::default(),
                );
                (real_targets, prediction_targets)
            }
        };
        let predictions = prediction_targets
            .random_like(Distribution::Uniform(0.0, THRESHOLD - 0.1))
            .sub(prediction_targets.clone())
            .abs();

        ClassificationInput::new(predictions, real_targets)
    }
}
