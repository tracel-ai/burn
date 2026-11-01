use super::scatter::adds_atomically;
use crate::kernel::{
    AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp, OrOp,
    utils::{address_type, shape_divmod},
};
use crate::tensor::CubeTensor;
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{CubeDim, calculate_cube_count_elemwise, std::tensor::layout::linear::LinearView};
use cubecl::{prelude::*, std::FastDivmod};

/// Uses checked launch mode because user-provided `indices` may contain out-of-bounds values
/// that would cause invalid writes into `tensor`. Checked mode clamps these accesses rather
/// than producing undefined behavior.
#[cube(launch, address_type = "dynamic")]
fn select_assign_kernel<F: Numeric, I: Numeric, Op: BinaryOpFamily>(
    tensor: &mut Tensor<F>,
    indices: LinearView<'_, I>,
    value: &Tensor<F>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let rank = value_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_tensor = 0;
    let mut offset_value = 0;

    // Calculate offsets and num_elems
    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != axis {
            let (rem, local_pos) = value_shape[i].div_mod(offset);
            offset = rem;

            offset_tensor += local_pos * tensor.stride(i);
            offset_value += local_pos * value.stride(i);
        }
    }

    let strides_tensor_dim = tensor.stride(axis);
    let strides_value_dim = value.stride(axis);

    // Main operation
    for i in 0..value.shape(axis) {
        let index_tensor = usize::cast_from(indices.read(i)) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        let value = Op::BinaryOp::<F, Const<1>>::execute(
            Vector::cast_from(tensor[index_tensor]),
            Vector::cast_from(value[index_value]),
        );
        write_checked(tensor.as_mut_slice(), index_tensor, F::cast_from(value));
    }
}

/// One unit per source value, with independent source and accumulation dtypes.
/// Unlike scatter, select reads one index along the axis, avoiding per-dimension
/// index addressing. CUDA f32 atomic adds flush subnormal inputs and results to zero.
#[cube(launch_unchecked, address_type = "dynamic")]
fn select_assign_add_atomic_kernel<T: Numeric, V: Numeric, I: Numeric>(
    tensor: &mut Tensor<Atomic<T>>,
    indices: LinearView<'_, I>,
    value: &Tensor<V>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(T, V, I)] _dtypes: [ElemType; 3],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let rank = value_shape.len().comptime();
    let mut offset = ABSOLUTE_POS;
    let mut offset_tensor = 0;
    let mut offset_value = 0;
    let mut coord_axis = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        let (rem, local_pos) = value_shape[i].div_mod(offset);
        offset = rem;

        offset_value += local_pos * value.stride(i);
        if i == axis {
            coord_axis = local_pos;
        } else {
            offset_tensor += local_pos * tensor.stride(i);
        }
    }

    let index = usize::cast_from(indices.read(coord_axis));
    if index < tensor.shape(axis) {
        let target = offset_tensor + index * tensor.stride(axis);
        if target < tensor.buffer_len() {
            tensor[target].fetch_add(T::cast_from(value[offset_value]));
        }
    }
}

fn atomic_storage_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Flex32 => DType::F32,
        other => other,
    }
}

fn select_assign_add_atomic(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let working_units = value.meta.num_elements();
    if working_units == 0 {
        return tensor;
    }
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);
    let (tensor_dtype, indices_dtype) = (atomic_storage_dtype(tensor.dtype), indices.dtype);
    let value_dtype = atomic_storage_dtype(value.dtype);
    let shape = shape_divmod(&value);

    unsafe {
        select_assign_add_atomic_kernel::launch_unchecked(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, value),
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            value.into_tensor_arg(),
            shape,
            working_units,
            dim,
            [
                dtype_to_storage_type(tensor_dtype),
                dtype_to_storage_type(value_dtype),
                dtype_to_storage_type(indices_dtype),
            ],
        );
    }

    tensor
}

fn select_assign_op<Op: BinaryOpFamily>(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let working_units = tensor.meta.num_elements() / tensor.meta.shape()[dim];
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let shape = shape_divmod(&value);
    select_assign_kernel::launch::<Op>(
        &tensor.client,
        cube_count,
        cube_dim,
        address_type!(tensor, indices, value),
        tensor.clone().into_tensor_arg(),
        indices.into_linear_view(),
        value.into_tensor_arg(),
        shape,
        working_units,
        dim,
        [
            dtype_to_storage_type(tensor_dtype),
            dtype_to_storage_type(indices_dtype),
        ],
    );

    tensor
}

pub(crate) fn select_assign(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
    is_bool: bool,
) -> CubeTensor {
    match is_bool {
        true => select_assign_op::<OrOp>(tensor, dim, indices, value),
        // Prioritize f32 accumulation for half-precision dtypes even on devices that
        // advertise native BF16/F16 atomics, avoiding mantissa saturation on repeated indices.
        false
            if matches!(tensor.dtype, DType::BF16 | DType::F16)
                && adds_atomically(&tensor.client, DType::F32) =>
        {
            let orig_dtype = tensor.dtype;
            let tensor_f32 = crate::kernel::cast(tensor, DType::F32);
            let out_f32 = select_assign_add_atomic(tensor_f32, dim, indices, value);
            crate::kernel::cast(out_f32, orig_dtype)
        }
        false if adds_atomically(&tensor.client, atomic_storage_dtype(tensor.dtype)) => {
            select_assign_add_atomic(tensor, dim, indices, value)
        }
        false => select_assign_op::<AddOp>(tensor, dim, indices, value),
    }
}

pub(crate) fn select_assign_mul(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    select_assign_op::<MulOp>(tensor, dim, indices, value)
}

pub(crate) fn select_assign_replace(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    select_assign_op::<AssignOp>(tensor, dim, indices, value)
}

pub(crate) fn select_assign_min(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    select_assign_op::<BinaryMinOp>(tensor, dim, indices, value)
}

pub(crate) fn select_assign_max(
    tensor: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    select_assign_op::<BinaryMaxOp>(tensor, dim, indices, value)
}
