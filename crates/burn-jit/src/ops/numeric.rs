use crate::kernel::{
    launch_binop, launch_scalar_binop, AddOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
};
use crate::{element::JitElement, tensor::JitTensor};
use crate::{FloatElement, JitRuntime};
use burn_tensor::{ElementConversion, Shape};
use cubecl::client::ComputeClient;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{tensor_vectorization_factor, Runtime};

pub fn full<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    full_device::<R, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape<D>,
    device: R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let empty = empty_device(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric + Vectorized>(tensor: &mut Tensor<C>, value: C) {
        if ABSOLUTE_POS >= tensor.len() {
            return;
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &empty.shape.dims, &empty.strides, D - 1);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    full_kernel::launch::<E::Primitive, R>(
        &empty.client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(
            vectorization_factor,
            &empty.handle,
            &empty.strides,
            &empty.shape.dims,
        ),
        ScalarArg::new(value),
    );

    empty
}

pub fn zeros<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 0.elem())
}

pub fn ones<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    ones_device::<R, E, D>(client, device.clone(), shape)
}

pub fn ones_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new_contiguous(client, device, shape, buffer)
}

pub fn add<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    launch_binop::<D, R, E, AddOp>(lhs, rhs)
}

pub fn add_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    launch_scalar_binop::<D, R, E, AddOp>(lhs, rhs)
}

pub fn sub<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    launch_binop::<D, R, E, SubOp>(lhs, rhs)
}

pub fn sub_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    launch_scalar_binop::<D, R, E, SubOp>(lhs, rhs)
}

pub fn mul<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    launch_binop::<D, R, E, MulOp>(lhs, rhs)
}

pub fn mul_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    launch_scalar_binop::<D, R, E, MulOp>(lhs, rhs)
}

pub fn div<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    launch_binop::<D, R, E, DivOp>(lhs, rhs)
}

pub fn div_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    launch_scalar_binop::<D, R, E, DivOp>(lhs, rhs)
}

pub fn remainder_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    launch_scalar_binop::<D, R, E, RemainderOp>(lhs, rhs)
}

pub fn pow<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    launch_binop::<D, R, E, PowOp>(lhs, rhs)
}
