use crate::{
    CubeRuntime,
    kernel::utils::{broadcast_shape, linear_view, linear_view_ref},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn cross_kernel<E: Float>(
    lhs: &LinearView<Line<E>>,
    rhs: &LinearView<Line<E>>,
    output: &mut LinearView<Line<E>, ReadWrite>,
    #[define(E)] _dtype: StorageType,
) {
    // Each thread processes one 3-element vector
    let vector_idx = ABSOLUTE_POS;
    let base_pos = vector_idx * 3;

    if !output.is_in_bounds(base_pos) {
        terminate!();
    }

    // Extract vectors
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

pub(crate) fn cross<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    dim: usize,
) -> CubeTensor<R> {
    let ndims = lhs.shape.num_dims();

    // Validate that the cross dimension has size 3
    if lhs.shape[dim] != 3 || rhs.shape[dim] != 3 {
        panic!(
            "Cross product requires dimension {} to have size 3, but got {} and {}",
            dim, lhs.shape[dim], rhs.shape[dim]
        );
    }

    // For now, only support cross on the last dimension
    if dim != ndims - 1 {
        unimplemented!(
            "Cross product on non-last dimension not yet implemented for CubeCL backend"
        );
    }

    let output_shape = broadcast_shape(&[&lhs, &rhs]);

    // Since the cross dimension is forced to be size 3, line size would be restricted to 1 anyway
    let line_size = 1;

    let output = empty_device_dtype::<R>(
        lhs.client.clone(),
        lhs.device.clone(),
        output_shape.clone(),
        lhs.dtype,
    );

    // Number of vectors to process
    let num_vectors = output_shape.num_elements() / 3;

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_vectors, cube_dim);

    unsafe {
        cross_kernel::launch_unchecked::<R>(
            &lhs.client,
            cube_count,
            cube_dim,
            linear_view_ref(&lhs, &output, line_size),
            linear_view_ref(&rhs, &output, line_size),
            linear_view(&output, line_size),
            lhs.dtype.into(),
        );
    }

    output
}
