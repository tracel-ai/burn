use crate::codegen::dialect::gpu::{
    BinaryOperation, Elem, Item, Operation, UnaryOperation, Variable,
};
use crate::codegen::Compiler;
use crate::{binary, Runtime};
use crate::{element::WgpuElement, tensor::WgpuTensor, unary};
use burn_compute::client::ComputeClient;
use burn_tensor::{ElementConversion, Shape};

pub fn full<R: Runtime, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    value: E,
) -> WgpuTensor<R, E, D> {
    let client = R::client(device);

    full_device::<R, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<R: Runtime, E: WgpuElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape<D>,
    device: R::Device,
    value: E,
) -> WgpuTensor<R, E, D> {
    let empty = empty_device(client, device, shape);

    unary!(
        operation: |elem: Elem| Operation::AssignLocal(UnaryOperation {
            input: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: empty; value,
        elem: E
    )
}

pub fn zeros<R: Runtime, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> WgpuTensor<R, E, D> {
    let client = R::client(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<R: Runtime, E: WgpuElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> WgpuTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 0.elem())
}

pub fn ones<R: Runtime, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> WgpuTensor<R, E, D> {
    let client = R::client(device);

    ones_device::<R, E, D>(client, device.clone(), shape)
}

pub fn ones_device<R: Runtime, E: WgpuElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> WgpuTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<R: Runtime, E: WgpuElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> WgpuTensor<R, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(client, device, shape, buffer)
}

pub fn add<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: E,
) -> WgpuTensor<R, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: E,
) -> WgpuTensor<R, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: E,
) -> WgpuTensor<R, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: E,
) -> WgpuTensor<R, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn pow<R: Runtime, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<R, E, D>,
    rhs: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Powf(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}
