use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

use crate::{
    kernel::utils::{address_type, broadcast_shape},
    ops::{max_vector_size_many, numeric::empty_device_dtype},
    tensor::CubeTensor,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn mask_fill_kernel<T: Numeric, B: Int, N: Size>(
    input: LinearView<'_, Vector<T, N>>,
    mask: LinearView<'_, Vector<B, N>>,
    mut output: LinearViewMut<'_, Vector<T, N>>,
    value: InputScalar,
    #[define(T, B)] _dtypes: [ElemType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mask = Vector::cast_from(mask.read(ABSOLUTE_POS));
    let input = input.read(ABSOLUTE_POS);
    let value = Vector::new(value.get::<T>());

    output.write(ABSOLUTE_POS, select_many(mask, value, input));
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
pub fn mask_fill(
    input: CubeTensor,
    mask: CubeTensor,
    value: InputScalar,
    strategy: MaskFillStrategy,
    dtype_bool: DType,
) -> CubeTensor {
    let ndims = input.meta.num_dims();
    let out_shape = broadcast_shape(&[&input, &mask]);
    if out_shape.num_elements() == 0 {
        return empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            out_shape,
            input.dtype,
        );
    }

    let vector_size = max_vector_size_many(&[&input, &mask], ndims - 1);
    let working_units = out_shape.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    let output = match strategy {
        MaskFillStrategy::Readonly => empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            out_shape,
            input.dtype,
        ),
        MaskFillStrategy::Inplace => input.clone(),
    };

    let out_arg = match strategy {
        MaskFillStrategy::Readonly => output.clone().into_linear_view(),
        MaskFillStrategy::Inplace => output.as_linear_view_alias(0),
    };

    let at = address_type!(input, mask, output);
    let mask = mask.into_linear_view_like(&output);

    unsafe {
        mask_fill_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            at,
            vector_size,
            input.into_linear_view_like(&output),
            mask,
            out_arg,
            value,
            [
                dtype_to_storage_type(output.dtype),
                dtype_to_storage_type(dtype_bool),
            ],
        );
    }

    output
}
