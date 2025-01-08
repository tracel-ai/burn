use crate::{element::JitElement, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

use super::{sum_dim, ReduceStrategy};

/// Sum all elements in the input buffer.
pub fn sum<R: JitRuntime, E: JitElement>(
    input: JitTensor<R>,
    strategy: ReduceStrategy,
) -> JitTensor<R> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R> =
        JitTensor::new_contiguous(input.client, input.device, shape, input.handle, input.dtype);
    sum_dim::<R, E, E>(input, 0, strategy).unwrap()
}
