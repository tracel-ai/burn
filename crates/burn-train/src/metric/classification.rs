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
    fn from(input: ClassificationInput<B>) -> Self {
        (input.predictions, input.targets)
    }
}

/// Class Averaging types for Classification metrics.
#[derive(Copy, Clone, Default)]
pub enum ClassReduction {
    ///Computes the statistics over all classes before averaging
    #[default]
    Micro,
    ///Computes the statistics independently for each class before averaging
    Macro,
}
