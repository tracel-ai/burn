use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

use super::{prod_dim, ReduceStrategy};

/// Multiply all elements in the input buffer.
pub fn prod<R: JitRuntime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    strategy: ReduceStrategy,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> =
        JitTensor::new_contiguous(input.client, input.device, shape, input.handle);
    prod_dim(input, 0, strategy)
}
