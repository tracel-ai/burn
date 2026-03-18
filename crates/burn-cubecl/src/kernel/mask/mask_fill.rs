use burn_backend::{DType, TensorMetadata};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

use crate::{
    CubeRuntime,
    kernel::utils::address_type,
    ops::{max_vector_size_many, numeric::empty_device_dtype},
    tensor::CubeTensor,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn mask_fill_kernel<T: Numeric, B: Int, N: Size>(
    input: &LinearView<Vector<T, N>>,
    mask: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    value: InputScalar,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mask = Vector::cast_from(mask[ABSOLUTE_POS]);
    let input = input[ABSOLUTE_POS];
    let value = Vector::new(value.get::<T>());

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
    let ndims = input.meta.num_dims();
    let output = match strategy {
        MaskFillStrategy::Readonly => empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            input.shape(),
            input.dtype,
        ),
        MaskFillStrategy::Inplace => input.clone(),
    };

    let vector_size = max_vector_size_many(&[&input, &mask], ndims - 1);
    let working_units = input.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    let out_arg = match strategy {
        MaskFillStrategy::Readonly => output.clone().into_linear_view(),
        MaskFillStrategy::Inplace => output.as_linear_view_alias(0),
    };

    let at = address_type!(input, mask, output);
    let mask = mask.into_linear_view_like(&input);

    unsafe {
        mask_fill_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            at,
            vector_size,
            input.into_linear_view(),
            mask,
            out_arg,
            value,
            [output.dtype.into(), dtype_bool.into()],
        );
    }

    output
}
