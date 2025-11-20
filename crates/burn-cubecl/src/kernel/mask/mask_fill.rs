use burn_tensor::DType;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{scalar::InputScalar, tensor::layout::linear::LinearView},
};

use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size_many, numeric::empty_device_dtype},
    tensor::CubeTensor,
};

#[cube(launch_unchecked)]
fn mask_fill_kernel<T: Numeric, B: Int>(
    input: &LinearView<Line<T>>,
    mask: &LinearView<Line<B>>,
    output: &mut LinearView<Line<T>, ReadWrite>,
    value: InputScalar,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mask = Line::cast_from(mask[ABSOLUTE_POS]);
    let input = input[ABSOLUTE_POS];
    let value = Line::new(value.get::<T>());

    output[ABSOLUTE_POS] = select_many(mask, value, input);
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask fill kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskFillStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the input tensor inplace.
    Inplace,
}

/// Execute the mask fill kernel with the given strategy.
pub fn mask_fill<R: CubeRuntime>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: InputScalar,
    strategy: MaskFillStrategy,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let ndims = input.shape.num_dims();
    let output = match strategy {
        MaskFillStrategy::Readonly => empty_device_dtype::<R>(
            input.client.clone(),
            input.device.clone(),
            input.shape.clone(),
            input.dtype,
        ),
        MaskFillStrategy::Inplace => input.clone(),
    };

    let cube_dim = CubeDim::default();
    let line_size = max_line_size_many(&[&input, &mask], ndims - 1);
    let cube_count =
        calculate_cube_count_elemwise(input.shape.num_elements() / line_size as usize, cube_dim);

    let out_arg = match strategy {
        MaskFillStrategy::Readonly => linear_view(&output, line_size),
        MaskFillStrategy::Inplace => linear_view_alias(&output, line_size, 0),
    };

    unsafe {
        mask_fill_kernel::launch_unchecked::<R>(
            &input.client,
            cube_count,
            cube_dim,
            linear_view(&input, line_size),
            linear_view_ref(&mask, &input, line_size),
            out_arg,
            value,
            [output.dtype.into(), dtype_bool.into()],
        );
    }

    output
}
