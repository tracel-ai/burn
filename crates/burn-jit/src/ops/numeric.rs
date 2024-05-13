use crate::codegen::dialect::gpu::{BinaryOperator, Elem, Operator, Scope};
use crate::gpu::{UnaryOperator, Variable};
use crate::{binary, Runtime};
use crate::{element::JitElement, tensor::JitTensor, unary};
use burn_compute::client::ComputeClient;
use burn_tensor::{ElementConversion, Shape};

pub fn full<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    full_device::<R, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape<D>,
    device: R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let empty = empty_device(client, device, shape);

    unary!(
        operation: |scope: &mut Scope, elem: Elem, _position: Variable| Operator::Assign(UnaryOperator {
            input: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: empty; value,
        elem: E
    )
}

pub fn zeros<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 0.elem())
}

pub fn ones<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    ones_device::<R, E, D>(client, device.clone(), shape)
}

pub fn ones_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new(client, device, shape, buffer)
}

pub fn add<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Add(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Add(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Sub(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Sub(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Mul(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Mul(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Div(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Div(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn remainder_scalar<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    let shape = lhs.shape.clone();
    let device = lhs.device.clone();

    let rhs_tensor = full::<R, E, D>(shape, &device, rhs);

    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Remainder(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs_tensor,
        elem: E
    )
}

pub fn pow<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Powf(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}
