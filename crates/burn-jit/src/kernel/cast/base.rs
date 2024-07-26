use std::any::TypeId;

use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

use crate::{tensor::JitTensor, JitElement, JitRuntime};

#[cube(launch)]
fn cast_kernel<T1: Numeric, T2: Numeric>(input: &Tensor<T1>, output: &mut Tensor<T2>) {
    output[ABSOLUTE_POS] = T2::cast_from(input[ABSOLUTE_POS]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: JitRuntime, EI: JitElement, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, EI, D>,
) -> JitTensor<R, EO, D> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return JitTensor::new_contiguous(
            tensor.client,
            tensor.device,
            tensor.shape,
            tensor.handle,
        );
    }

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
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    cast_kernel::launch::<EI::Primitive, EO::Primitive, R>(
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
