use burn_tensor::DType;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

use crate::{
    CubeRuntime,
    kernel::utils::{broadcast_shape, linear_view, linear_view_alias, linear_view_ref},
    ops::{max_line_size_many, numeric::empty_device_dtype},
    tensor::CubeTensor,
};

#[cube(launch)]
fn mask_where_kernel<T: Numeric, B: Int>(
    input: &LinearView<Line<T>>,
    value: &LinearView<Line<T>>,
    mask: &LinearView<Line<B>>,
    output: &mut LinearView<Line<T>, ReadWrite>,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    let pos = ABSOLUTE_POS;
    if !output.is_in_bounds(pos) {
        terminate!();
    }

    output[pos] = select_many(Line::cast_from(mask[pos]), value[pos], input[pos]);
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
pub fn mask_where<R: CubeRuntime>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    strategy: MaskWhereStrategy,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let cube_dim = CubeDim::default();
    let line_size = max_line_size_many(&[&input, &mask, &value], input.shape.num_dims() - 1);
    let cube_count =
        calculate_cube_count_elemwise(input.shape.num_elements() / line_size as usize, cube_dim);

    let out_shape = broadcast_shape(&[&input, &mask, &value]);

    let output = match strategy {
        MaskWhereStrategy::Readonly => empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            out_shape,
            input.dtype,
        ),
        MaskWhereStrategy::InplaceLhs => input.clone(),
        MaskWhereStrategy::InplaceRhs => value.clone(),
    };

    let out = match strategy {
        MaskWhereStrategy::Readonly => linear_view(&output, line_size),
        MaskWhereStrategy::InplaceLhs => linear_view_alias(&output, line_size, 0),
        MaskWhereStrategy::InplaceRhs => linear_view_alias(&output, line_size, 1),
    };

    mask_where_kernel::launch(
        &input.client,
        cube_count,
        cube_dim,
        linear_view_ref(&input, &output, line_size),
        linear_view_ref(&value, &output, line_size),
        linear_view_ref(&mask, &output, line_size),
        out,
        [output.dtype.into(), dtype_bool.into()],
    )
    .expect("Kernel to never fail");

    output
}
