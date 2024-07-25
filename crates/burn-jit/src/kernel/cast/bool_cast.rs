use crate::{tensor::JitTensor, JitElement, JitRuntime};
use cubecl::{calculate_cube_count_elemwise, cpa, prelude::*, CubeDim};

#[cube(launch)]
fn bool_cast_kernel<T: Numeric>(input: &Tensor<Bool>, output: &mut Tensor<T>) {
    output[ABSOLUTE_POS] = T::cast_from(input[ABSOLUTE_POS]);
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: JitRuntime, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, u32, D>,
) -> JitTensor<R, EO, D> {
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let vectorization = vectorization(num_elems);
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim.x as usize);

    bool_cast_kernel::launch::<EO::Primitive, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(
            vectorization,
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        ),
        TensorArg::vectorized(
            vectorization,
            &output.handle,
            &output.strides,
            &output.shape.dims,
        ),
    );

    output
}
