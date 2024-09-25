use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

use super::{sum_dim, ReduceStrategy};

/// Sum all elements in the input buffer.
pub fn sum<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    strategy: ReduceStrategy,
) -> JitTensor<R, E> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E> =
        JitTensor::new_contiguous(input.client, input.device, shape, input.handle);
    sum_dim(input, 0, strategy)
}
