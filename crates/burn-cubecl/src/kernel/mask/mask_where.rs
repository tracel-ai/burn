use burn_backend::DType;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

use crate::{
    CubeRuntime,
    kernel::utils::{address_type, broadcast_shape},
    ops::{max_vector_size_many, numeric::empty_device_dtype},
    tensor::CubeTensor,
};

#[cube(launch, address_type = "dynamic")]
fn mask_where_kernel<T: Numeric, B: Int, N: Size>(
    input: &LinearView<Vector<T, N>>,
    value: &LinearView<Vector<T, N>>,
    mask: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    let pos = ABSOLUTE_POS;
    if !output.is_in_bounds(pos) {
        terminate!();
    }

    output[pos] = select_many(Vector::cast_from(mask[pos]), value[pos], input[pos]);
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
    let vector_size = max_vector_size_many(&[&input, &mask, &value], input.meta.num_dims() - 1);

    let working_units = input.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

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
        MaskWhereStrategy::Readonly => output.clone().into_linear_view(),
        MaskWhereStrategy::InplaceLhs => output.as_linear_view_alias(0),
        MaskWhereStrategy::InplaceRhs => output.as_linear_view_alias(1),
    };

    mask_where_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, value, mask, output),
        vector_size,
        input.into_linear_view_like(&output),
        value.into_linear_view_like(&output),
        mask.into_linear_view_like(&output),
        out,
        [output.dtype.into(), dtype_bool.into()],
    );

    output
}
