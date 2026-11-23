use crate::{
    CubeDevice,
    kernel::utils::{address_type, shape_divmod},
};
use crate::{element::CubeElement, tensor::CubeTensor};
use crate::{
    kernel::{
        AddOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
        launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int,
    },
    ops::max_vector_size,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, TensorMetadata};
use burn_std::Metadata;
use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, dialect::math::IsNanOp},
    prelude::*,
    std::tensor::layout::linear::LinearViewMut,
};
use cubecl::{client::Client, server::MemoryLayout};
use cubecl::{server::MemoryLayoutDescriptor, std::FastDivmod};

/// Creates a tensor filled with `value`
pub fn full<E: CubeElement>(shape: Shape, device: &CubeDevice, value: E) -> CubeTensor {
    let client = device.client();

    full_client::<E>(client, shape, device.clone(), value)
}

/// Creates a tensor filled with `value`
pub fn full_client<E: CubeElement>(
    client: Client,
    shape: Shape,
    device: CubeDevice,
    value: E,
) -> CubeTensor {
    let dtype = E::dtype();
    full_device_dtype(
        client,
        shape,
        device,
        InputScalar::new(value, dtype_to_storage_type(dtype)),
        dtype,
    )
}

/// Creates a tensor filled with `value`
pub fn full_device_dtype(
    client: Client,
    shape: Shape,
    device: CubeDevice,
    value: InputScalar,
    dtype: DType,
) -> CubeTensor {
    let empty = empty_device_dtype(client, device, shape, dtype);

    fill_device_dtype(empty, value)
}

/// Fills an existing tensor with `value`
pub(crate) fn fill_device_dtype(tensor: CubeTensor, value: InputScalar) -> CubeTensor {
    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn full_kernel<C: Numeric, N: Size>(
        mut tensor: LinearViewMut<'_, Vector<C, N>>,
        value: InputScalar,
        #[define(C)] _dtype: ElemType,
    ) {
        if !tensor.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }

        tensor.write(ABSOLUTE_POS, Vector::new(value.get::<C>()));
    }

    let num_elems = tensor.meta.num_elements();
    let vector_size = max_vector_size(&tensor);

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        full_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            address_type!(tensor),
            vector_size,
            tensor.clone().into_linear_view(),
            value,
            dtype_to_storage_type(tensor.dtype),
        );
    }

    tensor
}

/// Creates a tensor filled with zeros
pub fn zeros(device: CubeDevice, shape: Shape, dtype: DType) -> CubeTensor {
    let client = device.client();
    full_device_dtype(
        client,
        shape,
        device,
        InputScalar::new(0u32, dtype_to_storage_type(dtype)),
        dtype,
    )
}

/// Creates a tensor filled with ones
pub fn ones(device: CubeDevice, shape: Shape, dtype: DType) -> CubeTensor {
    let client = device.client();
    full_device_dtype(
        client,
        shape,
        device,
        InputScalar::new(1u32, dtype_to_storage_type(dtype)),
        dtype,
    )
}

/// Creates a tensor filled with zeros
pub fn zeros_client(client: Client, device: CubeDevice, shape: Shape, dtype: DType) -> CubeTensor {
    full_device_dtype(
        client,
        shape,
        device,
        InputScalar::new(0u32, dtype_to_storage_type(dtype)),
        dtype,
    )
}

/// Creates a tensor filled with ones
pub fn ones_client(client: Client, device: CubeDevice, shape: Shape, dtype: DType) -> CubeTensor {
    full_device_dtype(
        client,
        shape,
        device,
        InputScalar::new(1u32, dtype_to_storage_type(dtype)),
        dtype,
    )
}

/// Create a tensor with uninitialized memory
pub fn empty_device<E: CubeElement>(
    client: Client,
    device: CubeDevice,
    shape: Shape,
) -> CubeTensor {
    let MemoryLayout { memory, strides } = client.empty_tensor(shape.clone(), size_of::<E>());

    CubeTensor::new(
        client,
        memory,
        Metadata::new(shape, strides),
        device,
        E::dtype(),
    )
}

/// Create a tensor with uninitialized memory
pub fn empty_device_dtype(
    client: Client,
    device: CubeDevice,
    shape: Shape,
    dtype: DType,
) -> CubeTensor {
    let MemoryLayout { memory, strides } = client.empty_tensor(shape.clone(), dtype.size());

    CubeTensor::new(client, memory, Metadata::new(shape, strides), device, dtype)
}

/// Create a contiguous tensor with uninitialized memory
pub fn empty_device_contiguous_dtype(
    client: Client,
    device: CubeDevice,
    shape: Shape,
    dtype: DType,
) -> CubeTensor {
    let descriptor = MemoryLayoutDescriptor::contiguous(shape.clone(), dtype.size());
    let MemoryLayout { memory, strides } = client.empty_tensors(vec![descriptor]).remove(0);

    CubeTensor::new(client, memory, Metadata::new(shape, strides), device, dtype)
}

/// Add two tensors
pub fn add(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop::<AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop::<SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop::<MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop::<DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop::<RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop::<PowOp>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop_int::<BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop_int::<BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop_int::<BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop_int::<BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor(lhs: CubeTensor, rhs: CubeTensor) -> CubeTensor {
    launch_binop_int::<BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar(lhs: CubeTensor, rhs: InputScalar) -> CubeTensor {
    launch_scalar_binop_int::<BitwiseXorOp>(lhs, rhs)
}

/// Operation family trait for cumulative operations
pub(crate) trait CumulativeOpFamily: Send + Sync + 'static {
    type CumulativeOp<C: Numeric>: CumulativeOp<C>;
}

/// Trait for cumulative operations
#[cube]
pub(crate) trait CumulativeOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a cumulative operation
    fn execute(lhs: C, rhs: C) -> C;

    /// Get the initial value for the accumulator
    fn init_value(first_element: C) -> C;
}

// Operation types
struct SumOp;
struct ProdOp;
struct MaxOp;
struct MinOp;

// `N: Numeric` can't call the float-only `IsNan` trait after a comptime type check, so emit
// the same Cube IR operation directly. Callers keep this inside float-only comptime branches.
#[cube]
fn numeric_is_nan<N: Numeric>(value: N) -> bool {
    intrinsic!(|scope| {
        let is_nan = IsNanOp::new(scope.ctx_mut(), value.read_value(scope));
        scope.register_with_result(&is_nan).into()
    })
}

#[cube]
fn cumulative_max<N: Numeric>(lhs: N, rhs: N) -> N {
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        if numeric_is_nan::<N>(lhs) {
            lhs
        } else if numeric_is_nan::<N>(rhs) {
            rhs
        } else {
            max(lhs, rhs)
        }
    } else {
        max(lhs, rhs)
    }
}

#[cube]
fn cumulative_min<N: Numeric>(lhs: N, rhs: N) -> N {
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        if numeric_is_nan::<N>(lhs) {
            lhs
        } else if numeric_is_nan::<N>(rhs) {
            rhs
        } else {
            min(lhs, rhs)
        }
    } else {
        min(lhs, rhs)
    }
}

// Implement CumulativeOpFamily for each operation
impl CumulativeOpFamily for SumOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for ProdOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MaxOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MinOp {
    type CumulativeOp<C: Numeric> = Self;
}

// Implement CumulativeOp for each operation type
#[cube]
impl<N: Numeric> CumulativeOp<N> for SumOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs + rhs
    }

    fn init_value(_first_element: N) -> N {
        N::zero()
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for ProdOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs * rhs
    }

    fn init_value(_first_element: N) -> N {
        N::from_int(1)
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for MaxOp {
    fn execute(lhs: N, rhs: N) -> N {
        cumulative_max::<N>(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for MinOp {
    fn execute(lhs: N, rhs: N) -> N {
        cumulative_min::<N>(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

/// Generic cumulative operation kernel: one unit per line along `dim`, scanning it in order.
///
/// Each unit reads its line once and writes every prefix as it goes, so the work is linear in the
/// number of elements. The units run in parallel across lines, not along one: a tensor that is a
/// single long line runs on a single unit.
#[cube(launch_unchecked, address_type = "dynamic")]
fn cumulative_kernel<C: Numeric, O: CumulativeOpFamily>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(C)] _dtype: ElemType,
) {
    let length = input.shape(dim);
    if ABSOLUTE_POS >= output.len() / length {
        terminate!();
    }

    let rank = comptime![shape.len()];
    let mut remainder = ABSOLUTE_POS;
    let mut input_offset = 0;
    let mut output_offset = 0;

    #[unroll]
    for i in 0..shape.len() {
        let i = comptime![rank - i - 1];
        if i != dim {
            let (rem, local_idx) = shape.index(i).div_mod(remainder);
            remainder = rem;
            input_offset += local_idx * input.stride(i);
            output_offset += local_idx * output.stride(i);
        }
    }

    let input_stride = input.stride(dim);
    let output_stride = output.stride(dim);
    let mut result = O::CumulativeOp::<C>::init_value(input[input_offset]);
    for i in 0..length {
        result = O::CumulativeOp::<C>::execute(result, input[input_offset + i * input_stride]);
        output[output_offset + i * output_stride] = result;
    }
}

/// Compute the cumulative sum along a dimension
pub fn cumsum(input: CubeTensor, dim: usize) -> CubeTensor {
    cumulative_op::<SumOp>(input, dim)
}

/// Compute the cumulative product along a dimension
pub fn cumprod(input: CubeTensor, dim: usize) -> CubeTensor {
    cumulative_op::<ProdOp>(input, dim)
}

/// Compute the cumulative minimum along a dimension
pub fn cummin(input: CubeTensor, dim: usize) -> CubeTensor {
    cumulative_op::<MinOp>(input, dim)
}

/// Compute the cumulative maximum along a dimension
pub fn cummax(input: CubeTensor, dim: usize) -> CubeTensor {
    cumulative_op::<MaxOp>(input, dim)
}

/// Generic cumulative operation function
fn cumulative_op<O: CumulativeOpFamily>(input: CubeTensor, dim: usize) -> CubeTensor {
    let client = input.client.clone();
    let device = input.device.clone();

    let output = empty_device_dtype(client.clone(), device, input.shape(), input.dtype);

    // One unit per line along `dim`; an empty tensor has none.
    let working_units = output
        .meta
        .num_elements()
        .checked_div(input.shape()[dim])
        .unwrap_or(0);
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let shape = shape_divmod(&input);

    unsafe {
        cumulative_kernel::launch_unchecked::<O>(
            &client,
            cube_count,
            cube_dim,
            address_type!(input, output),
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape,
            dim,
            dtype_to_storage_type(output.dtype),
        );
    }

    output
}
