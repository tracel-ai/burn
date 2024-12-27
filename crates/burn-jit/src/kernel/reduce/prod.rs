use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

use super::{prod_dim, ReduceStrategy};

/// Multiply all elements in the input buffer.
pub fn prod<R: JitRuntime, E: JitElement>(
    input: JitTensor<R>,
    strategy: ReduceStrategy,
) -> JitTensor<R> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R> =
        JitTensor::new_contiguous(input.client, input.device, shape, input.handle, input.dtype);
    prod_dim::<R, E, E>(input, 0, strategy)
}
