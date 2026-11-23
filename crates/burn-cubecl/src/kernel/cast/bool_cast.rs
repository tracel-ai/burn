use crate::{
    kernel::utils::address_type,
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_std::DType;
use cubecl::{
    CubeDim, calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn bool_cast_kernel<B: Int, T: Numeric, N: Size>(
    input: LinearView<'_, Vector<B, N>>,
    mut output: LinearViewMut<'_, Vector<T, N>>,
    #[define(B, T)] _dtypes: [ElemType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output.write(
        ABSOLUTE_POS,
        Vector::cast_from(input.read(ABSOLUTE_POS) != Vector::zero()),
    );
}

/// Cast a bool tensor to the given element type.
///
/// Any nonzero U8 or U32 value represents true, so normalize to 0 or 1 before casting.
pub fn bool_cast(tensor: CubeTensor, out_dtype: DType) -> CubeTensor {
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        out_dtype,
    );

    let vector_size = max_vector_size(&tensor);
    let num_elems = tensor.meta.num_elements();
    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    let dtype = tensor.dtype;

    unsafe {
        bool_cast_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, output),
            vector_size,
            tensor.into_linear_view(),
            output.clone().into_linear_view(),
            [
                dtype_to_storage_type(dtype),
                dtype_to_storage_type(out_dtype),
            ],
        )
    };

    output
}
