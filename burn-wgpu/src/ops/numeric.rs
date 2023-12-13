use crate::codegen::{Elem, Operator, Variable};
use crate::compute::{compute_client, WgpuComputeClient};
use crate::kernel::{binary_elemwise_default, binary_elemwise_inplace_default};
use crate::{
    binary_elemwise, binary_elemwise_inplace, element::WgpuElement, tensor::WgpuTensor, unary,
};
use crate::{GraphicsApi, WgpuDevice};
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
        operator: |elem: Elem| Operator::AssignLocal {
            input: Variable::Scalar(0, elem),
            out: Variable::Local(0, elem),
        },
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

pub fn add<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise!(Add, "+");
    binary_elemwise_inplace!(AddInplace, "+");

    if lhs.can_mut_broadcast(&rhs) {
        return binary_elemwise_inplace_default::<AddInplace, E, D>(lhs, rhs);
    }

    if rhs.can_mut_broadcast(&lhs) {
        return binary_elemwise_inplace_default::<AddInplace, E, D>(rhs, lhs);
    }

    binary_elemwise_default::<Add, E, D>(lhs, rhs)
}

pub fn add_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operator::Add {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, elem),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise!(Sub, "-");
    binary_elemwise_inplace!(SubInplace, "-");

    if lhs.can_mut_broadcast(&rhs) {
        return binary_elemwise_inplace_default::<SubInplace, E, D>(lhs, rhs);
    }

    binary_elemwise_default::<Sub, E, D>(lhs, rhs)
}

pub fn sub_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operator::Sub {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, elem),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise!(Mul, "*");
    binary_elemwise_inplace!(MulInplace, "*");

    if lhs.can_mut_broadcast(&rhs) {
        return binary_elemwise_inplace_default::<MulInplace, E, D>(lhs, rhs);
    }

    if rhs.can_mut_broadcast(&lhs) {
        return binary_elemwise_inplace_default::<MulInplace, E, D>(rhs, lhs);
    }

    binary_elemwise_default::<Mul, E, D>(lhs, rhs)
}

pub fn mul_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operator::Mul {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, elem),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn div<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise!(Div, "/");
    binary_elemwise_inplace!(DivInplace, "/");

    if lhs.can_mut_broadcast(&rhs) {
        return binary_elemwise_inplace_default::<DivInplace, E, D>(lhs, rhs);
    }

    binary_elemwise_default::<Div, E, D>(lhs, rhs)
}

pub fn div_scalar<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<E, D> {
    unary!(
        operator: |elem: Elem| Operator::Div {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, elem),
        },
        input: lhs; rhs,
        elem: E
    )
}
