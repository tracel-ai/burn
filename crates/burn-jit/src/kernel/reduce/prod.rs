use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

use super::{prod_dim, ReduceStrategy};

/// Multiply all elements in the input buffer.
pub fn prod<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    strategy: ReduceStrategy,
) -> JitTensor<R, E> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E> =
        JitTensor::new_contiguous(input.client, input.device, shape, input.handle);
    prod_dim(input, 0, strategy)
}
