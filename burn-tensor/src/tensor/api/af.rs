use super::Tensor;
use crate::back::Backend;

pub fn relu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}
