use crate::backend::Backend;
use crate::{Tensor, activation};

/// Computes the log softmax cross entropy between logits and target probabilities.
///
/// # Arguments
///
/// * `logits` - The logits.
/// * `target_probs` - The target probabilities.
///
/// # Returns
///
/// The log softmax cross entropy.
pub fn cross_entropy_with_logits<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    target_probs: Tensor<B, D>,
) -> Tensor<B, 1> {
    let tensor = activation::log_softmax(logits, D - 1);
    let tensor = tensor.mul(target_probs);
    let tensor = tensor.sum_dim(D - 1);

    tensor.mean().neg()
}
