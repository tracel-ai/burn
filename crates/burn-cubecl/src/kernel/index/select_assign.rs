use crate::kernel::{AddOp, BinaryOp, BinaryOpFamily, OrOp, into_contiguous};
use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

#[cube(launch_unchecked)]
fn select_assign_kernel<F: Numeric, I: Numeric, Op: BinaryOpFamily>(
    tensor: &mut Tensor<F>,
    indices: &Tensor<I>,
    value: &Tensor<F>,
    dim: &u32,
    #[define(F, I)] _dtypes: [StorageType; 2],
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

        let value = Op::BinaryOp::<F>::execute(
            Line::cast_from(tensor[index_tensor]),
            Line::cast_from(value[index_value]),
        );
        tensor[index_tensor] = F::cast_from(value);
    }
}

pub(crate) fn select_assign<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
    is_bool: bool,
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let indices = into_contiguous(indices);

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
            num_elems *= tensor.shape[index];
        });
    let working_units = num_elems;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let launch = match is_bool {
        true => select_assign_kernel::launch_unchecked::<OrOp, R>,
        false => select_assign_kernel::launch_unchecked::<AddOp, R>,
    };

    unsafe {
        launch(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            // Ignored shape + custom strides.
            TensorArg::from_raw_parts_and_size(
                &indices.handle,
                &strides,
                &strides,
                1,
                indices.dtype.size(),
            ),
            value.as_tensor_arg(1),
            ScalarArg::new(dim as u32),
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    };

    tensor
}
