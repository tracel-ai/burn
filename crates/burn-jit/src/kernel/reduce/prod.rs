use crate::{element::JitElement, tensor::JitTensor, Runtime};
use burn_tensor::Shape;

use super::{prod_dim, ReduceStrategy};

/// Multiply all elements in the input buffer.
pub fn prod<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    strategy: ReduceStrategy,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> = JitTensor::new(input.client, input.device, shape, input.handle);
    prod_dim(input, 0, strategy)
}
