use crate::{
    binary,
    codegen::dialect::gpu::{BinaryOperation, Elem, Item, Operation, Variable},
    element::WgpuElement,
    kernel::StaticKernelSource,
    kernel::{binary::binary, unary::unary},
    tensor::WgpuTensor,
    unary, JitGpuBackend,
};
use std::mem;

macro_rules! comparison {
    (
        binary: $ops:expr,
        backend: $backend:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operation: $ops, compiler: <$backend as JitGpuBackend>::Compiler, elem_in: $elem, elem_out: $elem);

        launch_binary::<
            Ops<<$backend as JitGpuBackend>::Compiler, E, u32>,
            OpsInplaceLhs<<$backend as JitGpuBackend>::Compiler, E, u32>,
            OpsInplaceRhs<<$backend as JitGpuBackend>::Compiler, E, u32>,
            $backend,
            E,
            D
        >($lhs, $rhs)
    }};

    (
        unary: $ops:expr,
        backend: $backend:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        unary!(operation: $ops, compiler: <$backend as JitGpuBackend>::Compiler, scalar 1);

        launch_unary::<
            Ops<<$backend as JitGpuBackend>::Compiler, E>,
            OpsInplace<<$backend as JitGpuBackend>::Compiler, E>,
            $backend,
            E,
            D
        >($lhs, $rhs)
    }};
}

pub fn equal<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        binary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Input(1, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Equal(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Greater(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::Lower(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::GreaterEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<B: JitGpuBackend, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: E,
) -> WgpuTensor<B, u32, D> {
    comparison!(
        unary: |elem: Elem| Operation::LowerEqual(BinaryOperation {
            lhs: Variable::Input(0, Item::Scalar(elem)),
            rhs: Variable::Scalar(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(Elem::Bool)),
        }),
        backend: B,
        input: lhs; rhs,
        elem: E
    )
}

fn launch_binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, B: JitGpuBackend, E, const D: usize>(
    lhs: WgpuTensor<B, E, D>,
    rhs: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplaceLhs: StaticKernelSource,
    KernelInplaceRhs: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output = binary::<Kernel, KernelInplaceLhs, KernelInplaceRhs, B, E, D>(
        lhs,
        rhs,
        can_be_used_as_bool,
    );

    // We recast the tensor type.
    WgpuTensor::new(output.client, output.device, output.shape, output.handle)
}

fn launch_unary<Kernel, KernelInplace, B: JitGpuBackend, E, const D: usize>(
    tensor: WgpuTensor<B, E, D>,
    scalars: E,
) -> WgpuTensor<B, u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output =
        unary::<Kernel, KernelInplace, B, E, D>(tensor, Some(&[scalars]), can_be_used_as_bool);

    // We recast the tensor type.
    WgpuTensor::new(output.client, output.device, output.shape, output.handle)
}
