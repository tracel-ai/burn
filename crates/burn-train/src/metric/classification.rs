use burn_core::prelude::{Backend, Bool, Tensor};

/// Input for classification tasks.
#[derive(new, Debug, Clone)]
pub struct ClassificationInput<B: Backend> {
    /// Sample x Class Non thresholded normalized predictions.
    pub predictions: Tensor<B, 2>,
    /// Sample x Class one-hot encoded target.
    pub targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> From<ClassificationInput<B>> for (Tensor<B, 2>, Tensor<B, 2, Bool>) {
    fn from(val: ClassificationInput<B>) -> Self {
        (val.predictions, val.targets)
    }
}

/// Class Averaging types for Classification metrics.
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum ClassAverageType {
    ///Computes the statistics over all classes before averaging
    Micro,
    ///Computes the statistics independently for each class before averaging
    Macro,
}
