use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, shape_divmod},
};
use crate::{element::CubeElement, tensor::CubeTensor};
use crate::{
    kernel::{
        AddOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
        launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int,
    },
    ops::max_line_size,
};
use burn_backend::{DType, Shape};
use cubecl::std::{FastDivmod, tensor::layout::linear::LinearView};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{client::ComputeClient, server::Allocation};

/// Creates a tensor filled with `value`
pub fn full<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> CubeTensor<R> {
    let client = R::client(device);

    full_client::<R, E>(client, shape, device.clone(), value)
}

/// Creates a tensor filled with `value`
pub fn full_client<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> CubeTensor<R> {
    let dtype = E::dtype();
    full_device_dtype(client, shape, device, InputScalar::new(value, dtype), dtype)
}

/// Creates a tensor filled with `value`
pub fn full_device_dtype<R: CubeRuntime>(
    client: ComputeClient<R>,
    shape: Shape,
    device: R::Device,
    value: InputScalar,
    dtype: DType,
) -> CubeTensor<R> {
    let empty = empty_device_dtype(client, device, shape, dtype);

    #[cube(launch_unchecked)]
    pub fn full_kernel<C: Numeric>(
        tensor: &mut LinearView<C, ReadWrite>,
        value: InputScalar,
        #[define(C)] _dtype: StorageType,
    ) {
        if !tensor.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = value.get::<C>();
    }

    let num_elems = empty.shape.num_elements();
    let line_size = max_line_size(&empty);

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&empty.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&empty.client, working_units, cube_dim);

    unsafe {
        full_kernel::launch_unchecked(
            &empty.client,
            cube_count,
            cube_dim,
            linear_view(&empty, line_size),
            value,
            empty.dtype.into(),
        )
        .expect("Kernel to never fail");
    }

    empty
}

/// Creates a tensor filled with zeros
pub fn zeros<R: CubeRuntime>(device: R::Device, shape: Shape, dtype: DType) -> CubeTensor<R> {
    let client = R::client(&device);
    full_device_dtype(client, shape, device, InputScalar::new(0u32, dtype), dtype)
}

/// Creates a tensor filled with ones
pub fn ones<R: CubeRuntime>(device: R::Device, shape: Shape, dtype: DType) -> CubeTensor<R> {
    let client = R::client(&device);
    full_device_dtype(client, shape, device, InputScalar::new(1u32, dtype), dtype)
}

/// Creates a tensor filled with zeros
pub fn zeros_client<R: CubeRuntime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    full_device_dtype(client, shape, device, InputScalar::new(0u32, dtype), dtype)
}

/// Creates a tensor filled with ones
pub fn ones_client<R: CubeRuntime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    full_device_dtype(client, shape, device, InputScalar::new(1u32, dtype), dtype)
}

/// Create a tensor with uninitialized memory
pub fn empty_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let Allocation { handle, strides } = client.empty_tensor(&shape.dims, size_of::<E>());

    CubeTensor::new(client, handle, shape, device, strides, E::dtype())
}

/// Create a tensor with uninitialized memory
pub fn empty_device_dtype<R: CubeRuntime>(
    client: ComputeClient<R>,
    device: R::Device,
    shape: Shape,
    dtype: DType,
) -> CubeTensor<R> {
    let Allocation { handle, strides } = client.empty_tensor(&shape.dims, dtype.size());

    CubeTensor::new(client, handle, shape, device, strides, dtype)
}

/// Add two tensors
pub fn add<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop::<R, AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop::<R, SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop::<R, MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop::<R, DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop::<R, RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop::<R, PowOp>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop_int::<R, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop_int::<R, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: CubeTensor<R>) -> CubeTensor<R> {
    launch_binop_int::<R, BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar<R: CubeRuntime>(lhs: CubeTensor<R>, rhs: InputScalar) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, BitwiseXorOp>(lhs, rhs)
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
        N::from_int(0)
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
        max(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for MinOp {
    fn execute(lhs: N, rhs: N) -> N {
        min(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

/// Generic cumulative operation kernel
///
/// # Limitations
///
/// This is a **naive sequential implementation** along the cumulative dimension:
/// - Each output element sequentially reads all previous elements along the dimension
/// - Computational complexity: O(n^2) memory reads where n is the size of the cumulative dimension
/// - **Performance:** Suitable for small tensors or small dimensions. For large tensors,
///   performance will degrade significantly compared to an optimized parallel scan algorithm.
///
/// # TODO
///
/// Implement an efficient GPU-optimized parallel scan algorithm.
#[cube(launch_unchecked)]
fn cumulative_kernel<C: Numeric, O: CumulativeOpFamily>(
    input: &Tensor<C>,
    output: &mut LinearView<C, ReadWrite>,
    shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(C)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![shape.len()];
    let dim_stride = input.stride(dim);

    let mut remainder = ABSOLUTE_POS;
    let mut offset = 0;
    let mut dim_idx = 0;

    #[unroll]
    for i in 0..shape.len() {
        let i = comptime![rank - i - 1];
        let (rem, local_idx) = shape.index(i).div_mod(remainder);
        remainder = rem;
        if i == dim {
            dim_idx = local_idx;
        } else {
            offset += local_idx * input.stride(i);
        }
    }

    // Read first element
    let first_read_idx = offset + dim_idx * dim_stride;
    let first_elem = input[first_read_idx];

    // Initialize accumulator
    let mut result = O::CumulativeOp::<C>::init_value(first_elem);

    // Accumulate values
    for i in 0..=dim_idx {
        let read_idx = offset + i * dim_stride;
        result = O::CumulativeOp::<C>::execute(result, input[read_idx]);
    }
    output[ABSOLUTE_POS] = result;
}

/// Compute the cumulative sum along a dimension
pub fn cumsum<R: CubeRuntime>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, SumOp>(input, dim)
}

/// Compute the cumulative product along a dimension
pub fn cumprod<R: CubeRuntime>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, ProdOp>(input, dim)
}

/// Compute the cumulative minimum along a dimension
pub fn cummin<R: CubeRuntime>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, MinOp>(input, dim)
}

/// Compute the cumulative maximum along a dimension
pub fn cummax<R: CubeRuntime>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, MaxOp>(input, dim)
}

/// Generic cumulative operation function
fn cumulative_op<R: CubeRuntime, O: CumulativeOpFamily>(
    input: CubeTensor<R>,
    dim: usize,
) -> CubeTensor<R> {
    let client = input.client.clone();
    let device = input.device.clone();

    let output = empty_device_dtype(client.clone(), device, input.shape.clone(), input.dtype);

    let num_elems = output.shape.num_elements();
    let working_units = num_elems;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);

    unsafe {
        cumulative_kernel::launch_unchecked::<O, R>(
            &client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            linear_view(&output, 1),
            shape_divmod(&input),
            dim,
            output.dtype.into(),
        )
        .expect("Kernel to never fail");
    }

    output
}
