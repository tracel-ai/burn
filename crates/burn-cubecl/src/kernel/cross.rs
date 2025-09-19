use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::utils::{broadcast_shape, linear_view, linear_view_ref},
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn cross_kernel<E: Float>(
    lhs: &LinearView<Line<E>>,
    rhs: &LinearView<Line<E>>,
    output: &mut LinearView<Line<E>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    // For now, assume the last dimension has size 3 and contains the vectors
    // Each ABSOLUTE_POS corresponds to the first element of a 3-element vector
    let base_pos = ABSOLUTE_POS * 3;

    // Extract vectors - Line<F> can be used directly as scalar
    let a0 = lhs[base_pos];
    let a1 = lhs[base_pos + 1];
    let a2 = lhs[base_pos + 2];
    let b0 = rhs[base_pos];
    let b1 = rhs[base_pos + 1];
    let b2 = rhs[base_pos + 2];

    // Compute cross product: a × b
    let x = a1 * b2 - a2 * b1;
    let y = a2 * b0 - a0 * b2;
    let z = a0 * b1 - a1 * b0;

    // Store result
    output[base_pos] = x;
    output[base_pos + 1] = y;
    output[base_pos + 2] = z;
}

const VECTOR_SIZE: u8 = 3;

pub(crate) fn cross<R: CubeRuntime, E: CubeElement + Float>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dim: usize,
) -> CubeTensor<R> {
    let ndims = lhs.shape.num_dims();

    // Validate that the cross dimension has size 3
    if lhs.shape.dims[dim] != 3 || rhs.shape.dims[dim] != 3 {
        panic!(
            "Cross product requires dimension {} to have size 3, but got {} and {}",
            dim, lhs.shape.dims[dim], rhs.shape.dims[dim]
        );
    }

    // For now, only support cross on the last dimension
    if dim != ndims - 1 {
        unimplemented!(
            "Cross product on non-last dimension not yet implemented for CubeCL backend"
        );
    }

    let output_shape = broadcast_shape(&[&lhs, &rhs]);
    let output = empty_device::<R, E>(lhs.client.clone(), lhs.device.clone(), output_shape.clone());

    // Number of 3-element vectors to process
    let num_vectors = output_shape.num_elements() / 3;

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_vectors, cube_dim);

    unsafe {
        cross_kernel::launch_unchecked::<E, R>(
            &lhs.client,
            cube_count,
            cube_dim,
            linear_view(&lhs, &VECTOR_SIZE),
            linear_view_ref(&rhs, &lhs, &VECTOR_SIZE),
            linear_view(&output, &VECTOR_SIZE),
        );
    }

    output
}
