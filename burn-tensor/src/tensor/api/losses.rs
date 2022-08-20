use super::{af, Tensor};
use crate::back::Backend;

pub fn cross_entropy_with_logits<B: Backend, const D: usize>(
    logits: &Tensor<B, D>,
    target_probs: &Tensor<B, D>,
) -> Tensor<B, 1> {
    let tensor = af::log_softmax(logits, D - 1);
    let tensor = tensor.mul(&target_probs);
    let tensor = tensor.sum_dim(D - 1);

    tensor.mean().neg()
}
