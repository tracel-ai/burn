use crate::codegen::dialect::gpu::{
    BinaryOperation, Elem, Item, Operation, UnaryOperation, Variable,
};
use crate::codegen::Compiler;
use crate::compute::{compute_client, WgpuComputeClient};
use crate::{binary, GraphicsApi, WgpuDevice};
use crate::{element::WgpuElement, tensor::WgpuTensor, unary};
use burn_tensor::{Element, ElementConversion, Shape};

pub fn full<G: GraphicsApi, E: WgpuElement + Element, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    value: E,
) -> WgpuTensor<E, D> {
    let client = compute_client::<G>(device);

    full_device(client, shape, device.clone(), value)
}

pub fn full_device<E: WgpuElement + Element, const D: usize>(
    client: WgpuComputeClient,
    shape: Shape<D>,
    device: WgpuDevice,
    value: E,
) -> WgpuTensor<E, D> {
    let empty = empty_device(client, device, shape);

    unary!(
        operator: |elem: Elem| Operation::AssignLocal(UnaryOperation {
            input: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        input: empty; value,
        elem: E
    )
}

pub fn zeros<G: GraphicsApi, E: WgpuElement + Element, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    let client = compute_client::<G>(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<E: WgpuElement + Element, const D: usize>(
    client: WgpuComputeClient,
    device: WgpuDevice,
    shape: Shape<D>,
) -> WgpuTensor<E, D> {
    full_device::<E, D>(client, shape, device, 0.elem())
}

pub fn ones<G: GraphicsApi, E: WgpuElement + Element, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    let client = compute_client::<G>(device);

    ones_device(client, device.clone(), shape)
}

pub fn ones_device<E: WgpuElement + Element, const D: usize>(
    client: WgpuComputeClient,
    device: WgpuDevice,
    shape: Shape<D>,
) -> WgpuTensor<E, D> {
    full_device::<E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<E: WgpuElement, const D: usize>(
    client: WgpuComputeClient,
    device: WgpuDevice,
    shape: Shape<D>,
) -> WgpuTensor<E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(client, device, shape, buffer)
}

pub fn add<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary!(
        operator: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary!(
        operator: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary!(
        operator: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary!(
        operator: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn pow<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary!(
        operator: |elem: Elem| Operation::Powf(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}
