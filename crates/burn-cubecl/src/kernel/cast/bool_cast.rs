use crate::{
    CubeRuntime,
    kernel::utils::address_type,
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use burn_std::DType;
use cubecl::{
    CubeDim, calculate_cube_count_elemwise, num_traits::One, prelude::*,
    std::tensor::layout::linear::LinearView,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn bool_cast_kernel<B: Int, T: Numeric, N: Size>(
    input: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    #[define(B, T)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS] & Vector::one());
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32 or u8
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: CubeRuntime>(tensor: CubeTensor<R>, out_dtype: DType) -> CubeTensor<R> {
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
            [dtype.into(), out_dtype.into()],
        )
    };

    output
}
