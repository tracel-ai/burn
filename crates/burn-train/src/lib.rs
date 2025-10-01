#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

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

mod evaluator;

pub use evaluator::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(test)]
pub(crate) mod tests {
    use crate::TestBackend;
    use burn_core::{prelude::Tensor, tensor::Bool};
    use std::default::Default;

    /// Probability of tp before adding errors
    pub const THRESHOLD: f64 = 0.5;

    #[derive(Debug, Default)]
    pub enum ClassificationType {
        #[default]
        Binary,
        Multiclass,
        Multilabel,
    }

    /// Sample x Class shaped matrix for use in
    /// classification metrics testing
    pub fn dummy_classification_input(
        classification_type: &ClassificationType,
    ) -> (Tensor<TestBackend, 2>, Tensor<TestBackend, 2, Bool>) {
        match classification_type {
            ClassificationType::Binary => {
                (
                    Tensor::from_data([[0.3], [0.2], [0.7], [0.1], [0.55]], &Default::default()),
                    // targets
                    Tensor::from_data([[0], [1], [0], [0], [1]], &Default::default()),
                    // predictions @ threshold=0.5
                    //                     [[0], [0], [1], [0], [1]]
                )
            }
            ClassificationType::Multiclass => {
                (
                    Tensor::from_data(
                        [
                            [0.2, 0.8, 0.0],
                            [0.3, 0.6, 0.1],
                            [0.7, 0.25, 0.05],
                            [0.1, 0.15, 0.8],
                            [0.9, 0.03, 0.07],
                        ],
                        &Default::default(),
                    ),
                    Tensor::from_data(
                        // targets
                        [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
                        // predictions @ top_k=1
                        //   [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0,  0]]
                        // predictions @ top_k=2
                        //   [[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0,  1]]
                        &Default::default(),
                    ),
                )
            }
            ClassificationType::Multilabel => {
                (
                    Tensor::from_data(
                        [
                            [0.1, 0.7, 0.6],
                            [0.3, 0.9, 0.05],
                            [0.8, 0.9, 0.4],
                            [0.7, 0.5, 0.9],
                            [1.0, 0.3, 0.2],
                        ],
                        &Default::default(),
                    ),
                    // targets
                    Tensor::from_data(
                        [[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]],
                        // predictions @ threshold=0.5
                        //   [[0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]]
                        &Default::default(),
                    ),
                )
            }
        }
    }
}
