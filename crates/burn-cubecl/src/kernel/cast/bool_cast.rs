use crate::{
    BoolElement, CubeElement, CubeRuntime,
    kernel::utils::linear_view,
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{
    CubeDim, calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView,
};

#[cube(launch_unchecked)]
fn bool_cast_kernel<B: Int, T: Numeric>(
    input: &LinearView<Line<B>>,
    output: &mut LinearView<Line<T>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(input[ABSOLUTE_POS] & Line::cast_from(1u32));
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32 or u8
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: CubeRuntime, BT: BoolElement, EO: CubeElement>(
    tensor: CubeTensor<R>,
) -> CubeTensor<R> {
    let output = empty_device::<R, EO>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );

    let line_size = max_line_size(&tensor);
    let num_elems = tensor.shape.num_elements();
    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        bool_cast_kernel::launch_unchecked::<BT, EO, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            linear_view(&tensor, line_size),
            linear_view(&output, line_size),
        )
        .expect("Kernel to never fail");
    }

    output
}
