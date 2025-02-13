use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use cubecl::frontend::{Numeric, Tensor, ABSOLUTE_POS};
use cubecl::linalg::tensor::index_offset_with_layout;
use cubecl::CubeDim;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn gather_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<Line<T>>,
    indices: &Tensor<Line<I>>,
    output: &mut Tensor<Line<T>>,
    dim: &u32,
) {
    if ABSOLUTE_POS >= indices.len() {
        terminate!();
    }

    let index = indices[ABSOLUTE_POS];

    let stride = input.stride(*dim);
    let mut offset = u32::cast_from(index);
    offset *= stride;

    if *dim > 0 {
        let offset_before = index_offset_with_layout(input, output, ABSOLUTE_POS, 0, *dim, false);
        offset += offset_before;
    }

    let offset_after =
        index_offset_with_layout(input, output, ABSOLUTE_POS, *dim + 1, input.rank(), false);
    offset += offset_after;
    output[ABSOLUTE_POS] = input[offset];
}

pub(crate) fn gather<R: CubeRuntime, E: CubeElement, I: CubeElement>(
    dim: usize,
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let shape_output = indices.shape.clone();
    let total_elem = shape_output.num_elements();
    let output = empty_device::<R, E>(tensor.client.clone(), tensor.device.clone(), shape_output);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);
    unsafe {
        gather_kernel::launch_unchecked::<E, I, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            indices.as_tensor_arg::<I>(1),
            output.as_tensor_arg::<E>(1),
            ScalarArg::new(dim as u32),
        )
    }
    output
}
