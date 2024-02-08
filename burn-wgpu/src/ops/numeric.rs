use crate::codegen::dialect::gpu::{
    BinaryOperation, Elem, Item, Operation, UnaryOperation, Variable,
};
use crate::codegen::Compiler;
use crate::{binary, JitGpuBackend};
use crate::{element::WgpuElement, tensor::WgpuTensor, unary};
use burn_compute::client::ComputeClient;
use burn_tensor::{ElementConversion, Shape};

pub fn full<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
    value: E,
) -> WgpuTensor<B, E, D> {
    let client = B::client(device);

    full_device::<B, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    client: ComputeClient<B::Server, B::Channel>,
    shape: Shape<D>,
    device: B::Device,
    value: E,
) -> WgpuTensor<B, E, D> {
    let empty = empty_device(client, device, shape);

    unary!(
        operation: |elem: Elem| Operation::AssignLocal(UnaryOperation {
            input: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: empty; value,
        elem: E
    )
}

pub fn zeros<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
) -> WgpuTensor<B, E, D> {
    let client = B::client(device);

    zeros_device::<B, E, D>(client, device.clone(), shape)
}

pub fn zeros_device<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    client: ComputeClient<B::Server, B::Channel>,
    device: B::Device,
    shape: Shape<D>,
) -> WgpuTensor<B, E, D> {
    full_device::<B, E, D>(client, shape, device, 0.elem())
}

pub fn ones<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
) -> WgpuTensor<B, E, D> {
    let client = B::client(device);

    ones_device::<B, E, D>(client, device.clone(), shape)
}

pub fn ones_device<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    client: ComputeClient<B::Server, B::Channel>,
    device: B::Device,
    shape: Shape<D>,
) -> WgpuTensor<B, E, D> {
    full_device::<B, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    client: ComputeClient<B::Server, B::Channel>,
    device: B::Device,
    shape: Shape<D>,
) -> WgpuTensor<B, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(client, device, shape, buffer)
}

pub fn add<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Add(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Sub(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Mul(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, E, D> {
    unary!(
        operation: |elem: Elem| Operation::Div(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}

pub fn pow<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    binary!(
        operation: |elem: Elem| Operation::Powf(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: B::Compiler,
        input: lhs; rhs,
        elem: E
    )
}
