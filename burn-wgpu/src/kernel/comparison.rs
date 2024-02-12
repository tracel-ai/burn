use crate::{
    binary,
    codegen::dialect::gpu::{BinaryOperation, Elem, Item, Operation, Variable},
    element::JitElement,
    kernel::StaticKernelSource,
    kernel::{binary::binary, unary::unary},
    tensor::JitTensor,
    unary, Runtime,
};
use std::mem;

macro_rules! comparison {
    (
        binary: $ops:expr,
        runtime: $runtime:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, elem_in: $elem, elem_out: $elem);

        launch_binary::<
            Ops<<$runtime as Runtime>::Compiler, E, u32>,
            OpsInplaceLhs<<$runtime as Runtime>::Compiler, E, u32>,
            OpsInplaceRhs<<$runtime as Runtime>::Compiler, E, u32>,
            $runtime,
            E,
            D
        >($lhs, $rhs)
    }};

    (
        unary: $ops:expr,
        runtime: $runtime:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        unary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, scalar 1);

        launch_unary::<
            Ops<<$runtime as Runtime>::Compiler, E>,
            OpsInplace<<$runtime as Runtime>::Compiler, E>,
            $runtime,
            E,
            D
        >($lhs, $rhs)
    }};
}

pub fn equal<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

fn launch_binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, R: Runtime, E, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplaceLhs: StaticKernelSource,
    KernelInplaceRhs: StaticKernelSource,
    E: JitElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output = binary::<Kernel, KernelInplaceLhs, KernelInplaceRhs, R, E, D>(
        lhs,
        rhs,
        can_be_used_as_bool,
    );

    // We recast the tensor type.
    JitTensor::new(output.client, output.device, output.shape, output.handle)
}

fn launch_unary<Kernel, KernelInplace, R: Runtime, E, const D: usize>(
    tensor: JitTensor<R, E, D>,
    scalars: E,
) -> JitTensor<R, u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: JitElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output =
        unary::<Kernel, KernelInplace, R, E, D>(tensor, Some(&[scalars]), can_be_used_as_bool);

    // We recast the tensor type.
    JitTensor::new(output.client, output.device, output.shape, output.handle)
}
