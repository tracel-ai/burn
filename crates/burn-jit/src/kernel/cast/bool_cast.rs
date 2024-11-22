use crate::{ops::max_vectorization, tensor::JitTensor, BoolElement, JitElement, JitRuntime};
use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

#[cube(launch)]
fn bool_cast_kernel<B: Numeric, T: Numeric>(input: &Tensor<Line<B>>, output: &mut Tensor<Line<T>>) {
    let val = Line::<bool>::cast_from(input[ABSOLUTE_POS]);
    output[ABSOLUTE_POS] = select_many(val, Line::new(T::from_int(1)), Line::new(T::from_int(0)));
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32 or u8
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: JitRuntime, B: BoolElement, EO: JitElement>(
    tensor: JitTensor<R>,
) -> JitTensor<R> {
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        buffer,
        EO::dtype(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);
    let vectorization = max_vectorization(&tensor);

    bool_cast_kernel::launch::<B, EO, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        tensor.as_tensor_arg::<B>(vectorization),
        output.as_tensor_arg::<EO>(vectorization),
    );

    output
}
