use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{layout::linear::LinearTensorView, r#virtual::ReadWrite},
};

use crate::{
    BoolElement, CubeRuntime,
    element::CubeElement,
    kernel::utils::linear_tensor,
    ops::{into_data_sync, max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};

#[cube(launch)]
fn mask_where_readonly_kernel<T: CubePrimitive, B: Int>(
    input: &LinearTensorView<T>,
    mask: &LinearTensorView<B>,
    value: &LinearTensorView<T>,
    output: &mut LinearTensorView<T, ReadWrite>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mask = Line::cast_from(mask[ABSOLUTE_POS]);

    output[ABSOLUTE_POS] = select_many(mask, value[ABSOLUTE_POS], input[ABSOLUTE_POS]);
}

#[cube(launch)]
fn mask_where_inplace_kernel<T: CubePrimitive, B: Int>(
    input: &mut LinearTensorView<T, ReadWrite>,
    mask: &LinearTensorView<B>,
    value: &LinearTensorView<T>,
    reverse: B,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    input[ABSOLUTE_POS] = select(
        mask[ABSOLUTE_POS] != Line::new(reverse),
        value[ABSOLUTE_POS],
        input[ABSOLUTE_POS],
    );
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask where kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskWhereStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the lhs tensor inplace.
    InplaceLhs,
    /// Reuse the rhs tensor inplace.
    InplaceRhs,
}

/// Execute the mask where kernel with the given strategy.
pub fn mask_where<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    strategy: MaskWhereStrategy,
) -> CubeTensor<R> {
    match strategy {
        MaskWhereStrategy::Readonly => mask_where_readonly::<R, E, BT>(input, mask, value),
        MaskWhereStrategy::InplaceLhs => mask_where_inplace::<R, E, BT>(input, mask, value, false),
        MaskWhereStrategy::InplaceRhs => mask_where_inplace::<R, E, BT>(value, mask, input, true),
    }
}

fn mask_where_readonly<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let output = empty_device::<R, EI>(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);
    let line_size = max_line_size(&input);

    mask_where_readonly_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        linear_tensor(&input, &line_size),
        linear_tensor(&mask, &line_size),
        linear_tensor(&value, &line_size),
        linear_tensor(&output, &line_size),
    );

    output
}

fn mask_where_inplace<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    reverse: bool,
) -> CubeTensor<R> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);
    let line_size = max_line_size(&input);

    println!(
        "mask where input: {}",
        into_data_sync::<R, EI>(input.clone())
    );
    println!("mask where mask: {}", into_data_sync::<R, EM>(mask.clone()));
    println!(
        "mask where value: {}",
        into_data_sync::<R, EI>(value.clone())
    );
    println!("mask where reverse: {}", reverse);

    mask_where_inplace_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        linear_tensor(&input, &line_size),
        linear_tensor(&mask, &line_size),
        linear_tensor(&value, &line_size),
        ScalarArg::new(EM::new_bool(reverse)),
    );

    println!(
        "mask where output: {}",
        into_data_sync::<R, EI>(input.clone())
    );

    input
}
