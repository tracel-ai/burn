use super::GradientsParams;
use crate::module::AutodiffModule;
use crate::record::Record;
use crate::tensor::backend::AutodiffBackend;
use crate::LearningRate;

/// General trait to optimize [module](AutodiffModule).
pub trait Optimizer<M, B>: Send
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    /// Optimizer associative type to be used when saving and loading the state.
    type Record: Record<B>;

    /// Perform the optimizer step using the given learning rate and gradients.
    /// The updated module is returned.
    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M;

    /// Get the current state of the optimizer as a [record](Record).
    fn to_record(&self) -> Self::Record;

    /// Load the state of the optimizer as a [record](Record).
    fn load_record(self, record: Self::Record) -> Self;
}
