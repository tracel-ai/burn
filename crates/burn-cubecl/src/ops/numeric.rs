use crate::{CubeRuntime, FloatElement, IntElement};
use crate::{element::CubeElement, tensor::CubeTensor};
use crate::{
    kernel::{
        AddOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
        launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int,
    },
    tensor::elem_size,
};
use burn_tensor::{ElementConversion, Shape};
use cubecl::client::ComputeClient;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

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
    let empty = empty_device_contiguous::<R, E>(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric>(tensor: &mut Array<C>, value: C) {
        if ABSOLUTE_POS >= tensor.len() {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.handle.handle.size() as usize / empty.handle.elem_size;
    let vectorization_factor = *R::supported_line_sizes()
        .iter()
        .filter(|factor| num_elems % **factor as usize == 0)
        .max()
        .unwrap_or(&1);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    full_kernel::launch::<E, R>(
        &empty.client,
        cube_count,
        cube_dim,
        // SAFETY: The tensor is unused, and overwriting the zero-fill is fine since the values
        // there are undefined.
        unsafe { empty.as_full_array_arg::<E>(vectorization_factor) },
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

/// Create a tensor with uninitialized memory and potentially pitched strides
pub fn empty_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let dtype = E::dtype();
    let buffer = client.empty_tensor(shape.dims, elem_size(dtype));

    CubeTensor::new(client, buffer, device, dtype)
}

/// Create a tensor with uninitialized memory
pub fn empty_device_contiguous<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let dtype = E::dtype();
    let buffer = client.empty(shape.num_elements() * elem_size(dtype));

    CubeTensor::new_contiguous(client, device, shape, buffer, dtype)
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
