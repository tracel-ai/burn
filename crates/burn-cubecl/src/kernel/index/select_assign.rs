use crate::kernel::{
    AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp, OrOp,
    utils::{address_type, shape_divmod},
};
use crate::tensor::CubeTensor;
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use burn_backend::{DType, TensorMetadata};
use cubecl::{
    CubeDim, calculate_cube_count_elemwise, features::AtomicUsage, ir::Type,
    std::tensor::layout::linear::LinearView,
};
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

/// Parallel one-thread-per-value atomic scatter-add along `axis`.
#[cube(launch_unchecked, address_type = "dynamic")]
fn select_assign_add_atomic_kernel<T: Numeric, I: Numeric>(
    tensor: &mut Tensor<Atomic<T>>,
    indices: LinearView<'_, I>,
    value: &Tensor<T>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(T, I)] _dtypes: [ElemType; 2],
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
            tensor[target].fetch_add(value[offset_value]);
        }
    }
}

/// Parallel one-thread-per-value atomic `f32` scatter-add from any float dtype `V` (`bf16`, `f16`, `f32`).
#[cube(launch_unchecked, address_type = "dynamic")]
fn select_assign_add_atomic_f32_kernel<V: Float, I: Numeric>(
    tensor: &mut Tensor<Atomic<f32>>,
    indices: LinearView<'_, I>,
    value: &Tensor<V>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(V, I)] _dtypes: [ElemType; 2],
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
            tensor[target].fetch_add(f32::cast_from(value[offset_value]));
        }
    }
}

fn atomic_storage_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Flex32 => DType::F32,
        other => other,
    }
}

fn adds_atomically(tensor: &CubeTensor) -> bool {
    let properties = tensor.client.properties();
    if properties.hardware.num_cpu_cores.is_some() {
        return false;
    }
    properties
        .atomic_type_usage(Type::atomic(dtype_to_elem_type(atomic_storage_dtype(
            tensor.dtype,
        ))))
        .contains(AtomicUsage::Add)
}

fn adds_atomically_f32(tensor: &CubeTensor) -> bool {
    let properties = tensor.client.properties();
    if properties.hardware.num_cpu_cores.is_some() {
        return false;
    }
    properties
        .atomic_type_usage(Type::atomic(dtype_to_elem_type(DType::F32)))
        .contains(AtomicUsage::Add)
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
                dtype_to_storage_type(indices_dtype),
            ],
        );
    }

    tensor
}

fn select_assign_add_atomic_f32(
    tensor_f32: CubeTensor,
    dim: usize,
    indices: CubeTensor,
    value: CubeTensor,
) -> CubeTensor {
    let working_units = value.meta.num_elements();
    if working_units == 0 {
        return tensor_f32;
    }
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);
    let (value_dtype, indices_dtype) = (atomic_storage_dtype(value.dtype), indices.dtype);
    let shape = shape_divmod(&value);

    unsafe {
        select_assign_add_atomic_f32_kernel::launch_unchecked(
            &tensor_f32.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor_f32, indices, value),
            tensor_f32.clone().into_tensor_arg(),
            indices.into_linear_view(),
            value.into_tensor_arg(),
            shape,
            working_units,
            dim,
            [
                dtype_to_storage_type(value_dtype),
                dtype_to_storage_type(indices_dtype),
            ],
        );
    }

    tensor_f32
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
                && adds_atomically_f32(&tensor) =>
        {
            let orig_dtype = tensor.dtype;
            let tensor_f32 = crate::kernel::cast(tensor, DType::F32);
            let out_f32 = select_assign_add_atomic_f32(tensor_f32, dim, indices, value);
            crate::kernel::cast(out_f32, orig_dtype)
        }
        false if adds_atomically(&tensor) => select_assign_add_atomic(tensor, dim, indices, value),
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
