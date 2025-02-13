use crate::{element::CubeElement, tensor::CubeTensor, CubeRuntime};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch_unchecked)]
fn select_assign_kernel<F: Numeric, I: Numeric>(
    tensor: &mut Tensor<F>,
    indices: &Tensor<I>,
    value: &Tensor<F>,
    dim: &u32,
) {
    let dim = *dim;
    let mut offset_tensor = 0u32;
    let mut offset_value = 0u32;
    let mut num_elems = 1u32;

    // Calculate offsets and num_elems
    for i in 0..tensor.rank() {
        if i != dim {
            let shape_tensor = tensor.shape(i);

            num_elems *= shape_tensor;

            let ogwl = ABSOLUTE_POS / indices.stride(i);

            offset_tensor += ogwl % shape_tensor * tensor.stride(i);
            offset_value += ogwl % value.shape(i) * value.stride(i);
        }
    }

    if ABSOLUTE_POS >= num_elems {
        terminate!();
    }

    let strides_tensor_dim = tensor.stride(dim);
    let strides_value_dim = value.stride(dim);

    // Main operation
    for i in 0..value.shape(dim) {
        let index_tensor = u32::cast_from(indices[i]) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        tensor[index_tensor] += value[index_value];
    }
}

pub(crate) fn select_assign<R: CubeRuntime, E: CubeElement, I: CubeElement>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut strides = vec![0; ndims];
    let mut current = 1;
    let mut num_elems = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems *= tensor.shape.dims[index];
        });
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        select_assign_kernel::launch_unchecked::<E, I, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            // Ignored shape + custom strides.
            TensorArg::from_raw_parts::<I>(&indices.handle, &strides, &strides, 1),
            value.as_tensor_arg::<E>(1),
            ScalarArg::new(dim as u32),
        );
    };

    tensor
}
