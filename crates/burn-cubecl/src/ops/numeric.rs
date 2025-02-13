use crate::kernel::{
    launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int, AddOp,
    BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
};
use crate::{element::JitElement, tensor::JitTensor};
use crate::{FloatElement, IntElement, JitRuntime};
use burn_tensor::{ElementConversion, Shape};
use cubecl::client::ComputeClient;
use cubecl::tensor_vectorization_factor;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

/// Create a tensor filled with `value`
pub fn full<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> JitTensor<R> {
    let client = R::client(device);

    full_device::<R, E>(client, shape, device.clone(), value)
}

/// Create a tensor filled with `value`
pub fn full_device<R: JitRuntime, E: JitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> JitTensor<R> {
    let ndims = shape.num_dims();
    let empty = empty_device::<R, E>(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric>(tensor: &mut Tensor<C>, value: C) {
        if ABSOLUTE_POS >= tensor.len() {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &empty.shape.dims, &empty.strides, ndims - 1);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    full_kernel::launch::<E, R>(
        &empty.client,
        cube_count,
        cube_dim,
        empty.as_tensor_arg::<E>(vectorization_factor),
        ScalarArg::new(value),
    );

    empty
}

/// Create a tensor filled with zeros
pub fn zeros<R: JitRuntime, E: JitElement>(shape: Shape, device: &R::Device) -> JitTensor<R> {
    let client = R::client(device);

    zeros_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with zeros
pub fn zeros_device<R: JitRuntime, E: JitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    full_device::<R, E>(client, shape, device, 0.elem())
}

/// Create a tensor filled with ones
pub fn ones<R: JitRuntime, E: JitElement>(shape: Shape, device: &R::Device) -> JitTensor<R> {
    let client = R::client(device);

    ones_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with ones
pub fn ones_device<R: JitRuntime, E: JitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    full_device::<R, E>(client, shape, device, 1.elem())
}

/// Create a tensor with uninitialized memory
pub fn empty_device<R: JitRuntime, E: JitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new_contiguous(client, device, shape, buffer, E::dtype())
}

/// Add two tensors
pub fn add<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder<R: JitRuntime, E: JitElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    launch_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow<R: JitRuntime, E: FloatElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, PowOp<E>>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and<R: JitRuntime, E: IntElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    launch_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar<R: JitRuntime, E: IntElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or<R: JitRuntime, E: IntElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    launch_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar<R: JitRuntime, E: IntElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor<R: JitRuntime, E: IntElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    launch_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar<R: JitRuntime, E: IntElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}
