use crate::{
    binary,
    codegen::{Elem, Operator, Variable},
    element::WgpuElement,
    kernel::StaticKernelSource,
    kernel::{binary::binary, unary::unary},
    tensor::WgpuTensor,
    unary,
};
use std::mem;

macro_rules! comparison {
    (
        binary: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operator: $ops, elem_in: $elem, elem_out: $elem);

        launch_binary::<Ops<E, u32>, OpsInplaceLhs<E, u32>, OpsInplaceRhs<E, u32>, E, D>($lhs, $rhs)
    }};

    (
        unary: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops, scalar 1);

        launch_unary::<Ops<E>, OpsInplace<E>, E, D>($lhs, $rhs)
    }};
}

pub fn equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Equal {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Greater {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::GreaterEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Lower {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::LowerEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Equal {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Greater {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Lower {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::GreaterEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::LowerEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

fn launch_binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, E, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplaceLhs: StaticKernelSource,
    KernelInplaceRhs: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output =
        binary::<Kernel, KernelInplaceLhs, KernelInplaceRhs, E, D>(lhs, rhs, can_be_used_as_bool);

    // We recast the tensor type.
    WgpuTensor::new(output.client, output.device, output.shape, output.handle)
}

fn launch_unary<Kernel, KernelInplace, E, const D: usize>(
    tensor: WgpuTensor<E, D>,
    scalars: E,
) -> WgpuTensor<u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output =
        unary::<Kernel, KernelInplace, E, D>(tensor, Some(&[scalars]), can_be_used_as_bool);

    // We recast the tensor type.
    WgpuTensor::new(output.client, output.device, output.shape, output.handle)
}
