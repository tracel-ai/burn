use crate::{
    kernel::{
        AddOp, AssignOp, BinaryOp, BinaryOpFamily, MulOp, OrOp,
        utils::{address_type, shape_divmod},
    },
    tensor::CubeTensor,
};
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use cubecl::{CubeDim, calculate_cube_count_elemwise, features::AtomicUsage, ir::Type};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    input: &mut Tensor<T>,
    indices: &Tensor<I>,
    value: &Tensor<T>,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(T, I)] _dtypes: [ElemType; 2],
) {
    let rank = in_shape.len().comptime();
    let stride_input = input.stride(dim);
    let stride_value = value.stride(dim);
    let stride_indices = indices.stride(dim);
    let shape_value = value.shape(dim);

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_indices = 0;
    let mut offset_value = 0;
    let mut num_elems = 1;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != dim {
            let shape_input_loop = input.shape(i);

            let (rem, local_pos) = in_shape[i].div_mod(offset);
            offset = rem;

            offset_input += local_pos * input.stride(i);
            offset_indices += local_pos * indices.stride(i);
            offset_value += local_pos * value.stride(i);

            num_elems *= shape_input_loop;
        }
    }

    let should_stop = ABSOLUTE_POS >= num_elems;
    if should_stop {
        terminate!();
    }

    for i in 0..shape_value {
        let value_idx = (stride_value * i) + offset_value;
        let index_idx = (stride_indices * i) + offset_indices;

        let value = value[value_idx];
        let index = usize::cast_from(indices[index_idx]);

        let input_idx = (stride_input * index) + offset_input;

        let value = Op::BinaryOp::<T, Const<1>>::execute(
            Vector::cast_from(input[input_idx]),
            Vector::cast_from(value),
        );
        input[input_idx] = value.extract(0usize);
    }
}

/// One unit per value, each update an atomic add: the whole value tensor
/// scatters in parallel. [`scatter_kernel`] runs one unit per output row and
/// walks the scatter axis serially, which leaves the device nearly idle when
/// the values far outnumber the output rows — a histogram or a bincount, where
/// a few thousand rows receive millions of updates.
///
/// Only for `Add`, and only on types the device adds atomically. On floats the
/// order the adds land in is not fixed, so the rounding of a sum can differ
/// from one run to the next; the serial kernel is deterministic.
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_add_atomic_kernel<T: Numeric, I: Int>(
    input: &mut Tensor<Atomic<T>>,
    indices: &Tensor<I>,
    value: &Tensor<T>,
    value_shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(T, I)] _dtypes: [ElemType; 2],
) {
    let rank = value_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_indices = 0;
    let mut offset_value = 0;
    let mut num_elems = 1;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        let (rem, coordinate) = value_shape[i].div_mod(offset);
        offset = rem;

        offset_value += coordinate * value.stride(i);
        offset_indices += coordinate * indices.stride(i);
        if i != dim {
            offset_input += coordinate * input.stride(i);
        }
        num_elems *= value.shape(i);
    }

    if ABSOLUTE_POS >= num_elems {
        terminate!();
    }

    let index = usize::cast_from(indices[offset_indices]);
    let target = offset_input + index * input.stride(dim);
    input[target].fetch_add(value[offset_value]);
}

fn scatter_add_atomic(
    dim: usize,
    tensor: CubeTensor,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let working_units = value.meta.num_elements();
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);
    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);
    let value_shape = shape_divmod(&value);

    unsafe {
        scatter_add_atomic_kernel::launch_unchecked(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, value),
            tensor.clone().into_tensor_arg(),
            indices.into_tensor_arg(),
            value.into_tensor_arg(),
            value_shape,
            dim,
            [
                dtype_to_storage_type(tensor_dtype),
                dtype_to_storage_type(indices_dtype),
            ],
        )
    }
    tensor
}

fn adds_atomically(tensor: &CubeTensor) -> bool {
    tensor
        .client
        .properties()
        .atomic_type_usage(Type::atomic(dtype_to_elem_type(tensor.dtype)))
        .contains(AtomicUsage::Add)
}

fn scatter_op<Op: BinaryOpFamily>(
    dim: usize,
    tensor: CubeTensor,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let num_elems = tensor.meta.num_elements() / tensor.meta.shape()[dim];

    let working_units = num_elems;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        scatter_kernel::launch_unchecked::<Op>(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, value),
            tensor.clone().into_tensor_arg(),
            indices.into_tensor_arg(),
            value.into_tensor_arg(),
            shape_divmod(&tensor),
            dim,
            [
                dtype_to_storage_type(tensor_dtype),
                dtype_to_storage_type(indices_dtype),
            ],
        )
    }
    tensor
}

pub(crate) fn scatter(
    dim: usize,
    tensor: CubeTensor,
    indices: CubeTensor,
    value: CubeTensor,
    is_bool: bool,
) -> CubeTensor {
    match is_bool {
        true => scatter_op::<OrOp>(dim, tensor, indices, value),
        false if adds_atomically(&tensor) => scatter_add_atomic(dim, tensor, indices, value),
        false => scatter_op::<AddOp>(dim, tensor, indices, value),
    }
}

pub(crate) fn scatter_mul(
    dim: usize,
    tensor: CubeTensor,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    scatter_op::<MulOp>(dim, tensor, indices, value)
}

pub(crate) fn scatter_assign(
    dim: usize,
    tensor: CubeTensor,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    scatter_op::<AssignOp>(dim, tensor, indices, value)
}
