use crate::{
    binary,
    codegen::dialect::gpu::{BinaryOperation, Elem, Item, Operation, Variable},
    codegen::Compiler,
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
        compiler: $compiler:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operator: $ops, compiler: $compiler, elem_in: $elem, elem_out: $elem);

        launch_binary::<Ops<$compiler, E, u32>, OpsInplaceLhs<$compiler, E, u32>, OpsInplaceRhs<$compiler, E, u32>, E, D>($lhs, $rhs)
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

pub fn equal<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<C: Compiler, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        compiler: C,
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
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
