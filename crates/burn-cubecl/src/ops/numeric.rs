use crate::{CubeRuntime, FloatElement, IntElement, kernel::utils::linear_view};
use crate::{element::CubeElement, tensor::CubeTensor};
use crate::{
    kernel::{
        AddOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
        launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int,
    },
    ops::max_line_size,
};
use burn_tensor::{ElementConversion, Shape};
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{client::ComputeClient, server::Allocation};

/// Create a tensor filled with `value`
pub fn full<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> CubeTensor<R> {
    let client = R::client(device);

    full_device::<R, E>(client, shape, device.clone(), value)
}

/// Create a tensor filled with `value`
pub fn full_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> CubeTensor<R> {
    let empty = empty_device::<R, E>(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric>(tensor: &mut LinearView<C, ReadWrite>, value: C) {
        if !tensor.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let line_size = max_line_size(&empty);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    full_kernel::launch::<E, R>(
        &empty.client,
        cube_count,
        cube_dim,
        linear_view(&empty, &line_size),
        ScalarArg::new(value),
    );

    empty
}

/// Create a tensor filled with zeros
pub fn zeros<R: CubeRuntime, E: CubeElement>(shape: Shape, device: &R::Device) -> CubeTensor<R> {
    let client = R::client(device);

    zeros_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with zeros
pub fn zeros_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    full_device::<R, E>(client, shape, device, 0.elem())
}

/// Create a tensor filled with ones
pub fn ones<R: CubeRuntime, E: CubeElement>(shape: Shape, device: &R::Device) -> CubeTensor<R> {
    let client = R::client(device);

    ones_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with ones
pub fn ones_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    full_device::<R, E>(client, shape, device, 1.elem())
}

/// Create a tensor with uninitialized memory
pub fn empty_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    CubeTensor::new_contiguous(client, device, shape, buffer, E::dtype())
}

/// Create a tensor with uninitialized memory
pub fn empty_device_strided<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let Allocation { handle, strides } = client.empty_tensor(&shape.dims, size_of::<E>());

    CubeTensor::new(client, handle, shape, device, strides, E::dtype())
}

/// Add two tensors
pub fn add<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, PowOp<E>>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}

/// Compute the cumulative sum along a dimension
///
/// Note: This is a naive sequential implementation along the cumsum dimension.
/// TODO: Implement an efficient GPU-optimized parallel scan algorithm (cubecl-scan crate)
/// References:
/// - https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
/// - https://www.w3.org/TR/WGSL/#builtin-subgroupInclusiveAdd
pub fn cumsum<R: CubeRuntime, E: CubeElement>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    #[cube(launch)]
    fn cumsum_kernel<C: Numeric>(
        input: &Tensor<C>,
        output: &mut Tensor<C>,
        dim_stride: u32,
        #[comptime] dim_size: u32,
    ) {
        if ABSOLUTE_POS >= output.len() {
            terminate!();
        }

        let idx = ABSOLUTE_POS;

        // Compute components of the index
        let before_dim = idx / dim_stride;
        let after_dim = idx % dim_stride;

        // Compute how many strides along dim we are
        let dim_offset = (idx / dim_stride) % dim_size;

        // Compute cumulative sum
        let mut sum = C::from_int(0);
        for i in 0..dim_size {
            if i <= dim_offset {
                let read_idx =
                    (before_dim / dim_size) * (dim_size * dim_stride) + i * dim_stride + after_dim;
                sum += input[read_idx];
            }
        }

        output[idx] = sum;
    }

    let client = input.client.clone();
    let device = input.device.clone();
    let shape = input.shape.clone();
    let dim_size = shape.dims[dim];

    // Calculate stride for the cumsum dimension
    let dim_stride: usize = shape.dims[dim + 1..].iter().product();

    let output = empty_device::<R, E>(client.clone(), device, shape);

    let num_elems = output.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    cumsum_kernel::launch::<E, R>(
        &client,
        cube_count,
        cube_dim,
        unsafe {
            TensorArg::from_raw_parts::<E>(&input.handle, &input.strides, &input.shape.dims, 1)
        },
        unsafe {
            TensorArg::from_raw_parts::<E>(&output.handle, &output.strides, &output.shape.dims, 1)
        },
        ScalarArg::new(dim_stride as u32),
        dim_size as u32,
    );

    output
}
