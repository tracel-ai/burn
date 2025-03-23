use crate::{
    BoolElement, CubeElement, CubeRuntime,
    kernel::index_pitched,
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{
    CubeDim, calculate_cube_count_elemwise, linalg::tensor::index_offset_contiguous, prelude::*,
};

#[cube(launch)]
fn bool_cast_kernel<B: Numeric, T: Numeric>(
    input: &Tensor<Line<B>>,
    output: &mut Tensor<Line<T>>,
    #[comptime] rank: Option<u32>,
    #[comptime] pitched: bool,
) {
    let offset_out = ABSOLUTE_POS;

    if offset_out >= output.len() {
        terminate!();
    }

    let offset_in = index_offset_contiguous(input, offset_out, rank);
    let offset_out = index_pitched(output, offset_out, pitched);

    let val = Line::<bool>::cast_from(input[offset_in]);
    output[offset_out] = Line::cast_from(val);
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
    let num_elems = tensor.shape().num_elements();
    let output = empty_device::<R, EO>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape().clone(),
    );

    let line_size = max_line_size(&tensor);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    bool_cast_kernel::launch::<BT, EO, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        tensor.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        Some(tensor.rank() as u32),
        tensor.is_pitched() || output.is_pitched(),
    );

    output
}
