use crate::{element::JitElement, kernel::Kernel, tensor::JitTensor, JitRuntime};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch_unchecked)]
fn select_assign_kernel<F: Numeric, I: Numeric>(
    tensor: &mut Tensor<F>,
    indices: &Tensor<I>,
    value: &Tensor<F>,
    dim: &UInt,
) {
    let dim = *dim;
    let mut offset_tensor = UInt::new(0u32);
    let mut offset_value = UInt::new(0u32);
    let mut num_elems = UInt::new(1u32);

    // Calculate offsets and num_elems
    for i in range(0, tensor.rank(), Comptime::new(false)) {
        if i != dim {
            let shape_tensor = tensor.shape(i);

            num_elems *= shape_tensor;

            let ogwl = ABSOLUTE_POS / tensor.stride(i);

            offset_tensor += ogwl % tensor.shape(i) * tensor.stride(i);
            offset_value += ogwl % value.shape(i) * value.stride(i);
        }
    }

    if num_elems >= ABSOLUTE_POS {
        return;
    }

    // Main operation
    for i in range(0, value.shape(dim), Comptime::new(false)) {
        let index_tensor = UInt::cast_from(indices[i]) * tensor.stride(dim) + offset_tensor;
        let index_value = i * value.stride(dim) + offset_value;

        tensor[index_tensor] += value[index_value];
    }
}

pub(crate) fn select_assign<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut num_elems = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, _val)| {
            num_elems *= tensor.shape.dims[index];
        });

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        select_assign_kernel::launch_unchecked::<E::Primitive, I::Primitive, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            TensorArg::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            TensorArg::new(&indices.handle, &indices.strides, &indices.shape.dims),
            TensorArg::new(&value.handle, &value.strides, &value.shape.dims),
            ScalarArg::new(dim as u32),
        );
    };

    tensor
}
