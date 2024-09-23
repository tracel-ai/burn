use crate::{tensor::JitTensor, JitElement, JitRuntime};
use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

#[cube(launch)]
fn bool_cast_kernel<T: Numeric>(input: &Tensor<u32>, output: &mut Tensor<T>) {
    if input[ABSOLUTE_POS] >= 1 {
        output[ABSOLUTE_POS] = T::from_int(1);
    } else {
        output[ABSOLUTE_POS] = T::from_int(0);
    }
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: JitRuntime, EO: JitElement>(tensor: JitTensor<R, u32>) -> JitTensor<R, EO> {
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        buffer,
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    bool_cast_kernel::launch::<EO, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        tensor.as_tensor_arg(1),
        output.as_tensor_arg(1),
    );

    output
}
